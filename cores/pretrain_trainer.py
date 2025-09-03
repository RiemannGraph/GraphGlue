import torch
from cores.models import RPGraphFM
from data.data_loader import load_pretrain_single_graph_data, load_pretrain_multi_graph_data
from torch_geometric.loader import DataLoader, NeighborLoader
from utils.logger import create_logger
from utils.checkpoints import save_checkpoint, load_checkpoint, get_latest_checkpoint, cleanup_old_checkpoints
import os
from typing import List
from torch_geometric.transforms import RootedEgoNets, Compose
from data.data_transform import RenameFromRootedEgoNets
from utils.tools import format_time
import time
from datetime import timedelta
import gc
import warnings

warnings.filterwarnings("ignore")


class Pretrainer:
    def __init__(self, configs, logger=None):
        self.final_model_path = None
        assert len(configs.num_neighbors) >= configs.k_hops, "number of neighbor hops are not match!"
        self.configs = configs
        self.pretrain_single_graph_data = configs.pretrain_single_graph_data
        self.pretrain_multi_graph_data = configs.pretrain_multi_graph_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RPGraphFM(configs).to(self.device)
        self.logger = create_logger(configs.log_path) if logger is None else logger
        self.start_epoch = 0
        self.start_time = None
        self.epoch_times = []
        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.configs.lr_pretrain,
            weight_decay=self.configs.pretrain_weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.configs.pretrain_epochs,
            eta_min=self.configs.lr_pretrain * 0.01
        )

        # Resume checkpoint if you want
        if self.configs.resume_checkpoint:
            latest_check_path = get_latest_checkpoint(self.configs.checkpoint_dir)
            if latest_check_path:
                self.start_epoch = load_checkpoint(
                    filepath=latest_check_path,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                self.logger.info(f"Resumed training from epoch {self.start_epoch}")
            else:
                self.start_epoch = 0
                self.logger.info("No checkpoint found. Start from scratch.")
        else:
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.configs.pretrain_epochs):
            epoch_start_time = time.time()

            train_loss = self._train_epoch(optimizer, epoch)

            scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f'Epoch {epoch:03d}/{self.configs.pretrain_epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Time: {epoch_time:.2f}s | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            if (epoch + 1) % self.configs.save_interval == 0 or (epoch + 1) == self.configs.pretrain_epochs:
                checkpoint_path = os.path.join(
                    self.configs.checkpoint_dir,
                    f'pretrain_epoch_{epoch + 1}.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=checkpoint_path
                )

                # Optional
                cleanup_old_checkpoints(self.configs.checkpoint_dir, keep_last=5)

            if (epoch + 1) == self.configs.pretrain_epochs:
                final_model_path = os.path.join(
                    self.configs.checkpoint_dir,
                    'pretrain_final_model.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=final_model_path
                )
                self.logger.info(f'Saved final model: {final_model_path}')
                self.final_model_path = final_model_path

    @torch.no_grad()
    def register_from_loaders(self):
        if self.final_model_path:
            load_checkpoint(self.final_model_path, self.model,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        proto_z_list = []
        proto_z_tan_list = []

        num_node_level_datasets = len(self.pretrain_single_graph_data)
        if num_node_level_datasets > 0:
            for data_name in self.pretrain_single_graph_data:
                z_parts, z_tan_parts = [], []
                loader = self._create_node_loader(data_name)
                for data in loader:
                    data = data.to(self.model.device)
                    z, z_tan = self.model(data, data.batch_graph_nums)
                    z_parts.append(z[: loader.batch_size].cpu())
                    z_tan_parts.append(z_tan[: loader.batch_size].cpu())

                z_proto = torch.cat(z_parts, dim=0).mean(dim=0, keepdim=True)  # [1, d]
                z_tan_proto = torch.cat(z_tan_parts, dim=0).mean(dim=0, keepdim=True)
                proto_z_list.append(z_proto)
                proto_z_tan_list.append(z_tan_proto)

                del loader
                gc.collect()

        num_graph_level_datasets = len(self.pretrain_multi_graph_data)
        if num_graph_level_datasets > 0:
            for data_name in self.pretrain_multi_graph_data:
                z_parts, z_tan_parts = [], []
                loader = self._create_graph_loader(data_name)
                for data in loader:
                    data = data.to(self.device)
                    z, z_tan = self.model(data, data.batch_size)
                    z_parts.append(z.cpu())
                    z_tan_parts.append(z_tan.cpu())
                z_proto = torch.cat(z_parts, dim=0).mean(dim=0, keepdim=True)
                z_tan_proto = torch.cat(z_tan_parts, dim=0).mean(dim=0, keepdim=True)
                proto_z_list.append(z_proto)
                proto_z_tan_list.append(z_tan_proto)

                del loader
                gc.collect()

        if not proto_z_list:
            raise ValueError("No loaders provided or no data processed.")

        final_proto_z = torch.cat(proto_z_list, dim=0)  # [K, d]
        final_proto_z_tan = torch.cat(proto_z_tan_list, dim=0)  # [K, M, d]
        self.model.register_prototypes(final_proto_z, final_proto_z_tan)

        return self.model

    def _train_epoch(self, optimizer, epoch):
        if self.start_time is None:
            self.start_time = time.time()
        start_epoch_time = time.time()

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        # Phase 1: Intra Training
        loss_stats = self._train_node_level(optimizer, epoch)
        total_loss += loss_stats['loss']
        total_batches += loss_stats['batches']

        loss_stats = self._train_graph_level(optimizer, epoch)
        total_loss += loss_stats['loss']
        total_batches += loss_stats['batches']

        # Phase 2: Inter Loss
        if epoch % self.configs.inter_loss_interval == 0:
            inter_loss = self._compute_inter_loss(optimizer, epoch)
            if inter_loss is not None:
                total_loss += inter_loss
                total_batches += 1
        else:
            self.logger.info(f"Epoch {epoch} | Skip inter loss (interval={self.configs.inter_loss_interval})")

        # Log
        self._log_epoch_summary(epoch, start_epoch_time)
        self._update_epoch_time(epoch, start_epoch_time)

        return total_loss / total_batches

    def _train_node_level(self, optimizer, epoch):
        total_loss = 0.0
        total_batches = 0
        num_datasets = len(self.pretrain_single_graph_data)

        if num_datasets == 0:
            return {'loss': 0.0, 'batches': 0}

        for data_idx, data_name in enumerate(self.pretrain_single_graph_data):
            self.logger.info(f'Processing node loader {data_idx + 1}/{num_datasets}')
            node_loader = self._create_node_loader(data_name)
            dataset_len = len(node_loader)
            start_loader_time = time.time()

            for batch_idx, data in enumerate(node_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                z, z_tan = self.model(data, data.batch_graph_nums)
                intra_loss = self.model.loss(z, z_tan, data.origin_edge_index, node_loader.batch_size)
                intra_loss.backward()
                optimizer.step()

                total_loss += intra_loss.item()
                total_batches += 1

                if batch_idx % self.configs.log_interval == 0:
                    self._log_progress(
                        epoch=epoch,
                        loader_type="Node",
                        loader_idx=data_idx + 1,
                        batch_idx=batch_idx,
                        dataset_len=dataset_len,
                        loss=intra_loss.item(),
                        start_loader_time=start_loader_time,
                        batches_done=batch_idx + 1
                    )

            del node_loader
            torch.cuda.empty_cache()

        return {'loss': total_loss, 'batches': total_batches}

    def _train_graph_level(self, optimizer, epoch):
        total_loss = 0.0
        total_batches = 0
        num_datasets = len(self.pretrain_multi_graph_data)

        if num_datasets == 0:
            return {'loss': 0.0, 'batches': 0}

        for data_idx, data_name in enumerate(self.pretrain_multi_graph_data):
            self.logger.info(f'Processing graph loader {data_idx + 1}/{num_datasets}')
            graph_loader = self._create_graph_loader(data_name)
            dataset_len = len(graph_loader)
            start_loader_time = time.time()

            for batch_idx, data in enumerate(graph_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                z, z_tan = self.model(data, data.batch_size)
                edge_index, _ = self.model.knn_graph(z, self.configs.knn)
                intra_loss = self.model.loss(z, z_tan, edge_index)
                intra_loss.backward()
                optimizer.step()

                total_loss += intra_loss.item()
                total_batches += 1

                if batch_idx % self.configs.log_interval == 0:
                    self._log_progress(
                        epoch=epoch,
                        loader_type="Graph",
                        loader_idx=data_idx + 1,
                        batch_idx=batch_idx,
                        dataset_len=dataset_len,
                        loss=intra_loss.item(),
                        start_loader_time=start_loader_time,
                        batches_done=batch_idx + 1
                    )

            del graph_loader
            torch.cuda.empty_cache()

        return {'loss': total_loss, 'batches': total_batches}

    def _collect_all_embeddings_for_inter(self):
        all_z = []
        all_z_tan = []
        for data_name in self.pretrain_single_graph_data:
            node_loader = self._create_node_loader(data_name)
            z_list, z_tan_list = [], []
            for data in node_loader:
                data = data.to(self.device)
                z, z_tan = self.model(data, data.batch_graph_nums)
                g_z = z[:node_loader.batch_size].mean(dim=0, keepdim=True)  # [1, d]
                g_z_tan = z_tan[:node_loader.batch_size].mean(dim=0, keepdim=True)
                z_list.append(g_z)
                z_tan_list.append(g_z_tan)
            if z_list:
                all_z.append(torch.cat(z_list, dim=0))  # [num_graphs, d]
                all_z_tan.append(torch.cat(z_tan_list, dim=0))
            del node_loader
            torch.cuda.empty_cache()

        for data_name in self.pretrain_multi_graph_data:
            graph_loader = self._create_graph_loader(data_name)
            for batch in graph_loader:
                batch = batch.to(self.device)
                z, z_tan = self.model(batch, batch.batch_size)
                all_z.append(z)
                all_z_tan.append(z_tan)
            del graph_loader
            torch.cuda.empty_cache()

        if len(all_z) == 0:
            return None, None

        all_z = torch.cat(all_z, dim=0).to(self.device)
        all_z_tan = torch.cat(all_z_tan, dim=0).to(self.device)
        return all_z, all_z_tan

    def _compute_inter_loss(self, optimizer, epoch):
        all_z, all_z_tan = self._collect_all_embeddings_for_inter()
        if all_z is None:
            self.logger.warning("Not enough embeddings for inter-loss.")
            return None

        optimizer.zero_grad()
        edge_index, _ = self.model.knn_graph(all_z, self.configs.knn)
        inter_loss = self.model.loss(all_z, all_z_tan, edge_index)

        inter_loss.backward()
        optimizer.step()

        self.logger.info(f'Epoch {epoch} | Inter Loss: {inter_loss.item():.6f}')
        return inter_loss.item()

    def _log_progress(self, epoch, loader_type, loader_idx, batch_idx, dataset_len, loss, start_loader_time,
                      batches_done):
        current_time = time.time()

        batches_remaining = dataset_len - batches_done
        recent_avg_batch_time = (current_time - start_loader_time) / batches_done
        loader_remaining_time = recent_avg_batch_time * batches_remaining

        if len(self.epoch_times) == 0:
            elapsed_total = current_time - self.start_time
            avg_epoch_time = elapsed_total / (epoch + 1)
        else:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = max(0, self.configs.pretrain_epochs - (epoch + 1))

        if epoch == 0:
            total_remaining_time = None
        else:
            total_remaining_time = avg_epoch_time * remaining_epochs

        self.logger.info(
            f'Epoch {epoch} | {loader_type} Loader {loader_idx} | '
            f'Batch {batch_idx}/{dataset_len} | '
            f'Loss: {loss:.6f} | '
            f'Loader ETA: {format_time(loader_remaining_time)} | '
            f'Total ETA: {format_time(total_remaining_time)}'
        )

    def _log_epoch_summary(self, epoch, start_epoch_time):
        if len(self.epoch_times) == 0:
            avg_epoch_time = time.time() - self.start_time
        else:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        remaining_epochs = max(0, self.configs.pretrain_epochs - (epoch + 1))
        if epoch == 0:
            total_remaining_time = None
        else:
            total_remaining_time = avg_epoch_time * remaining_epochs

        epoch_duration = time.time() - start_epoch_time

        self.logger.info(
            f'Epoch {epoch} completed in {format_time(epoch_duration)}. '
            f'Estimated remaining training time: {format_time(total_remaining_time)} '
            f'({remaining_epochs} epochs left)'
        )

    def _update_epoch_time(self, epoch, start_epoch_time):
        expected_len = epoch
        if len(self.epoch_times) == expected_len:
            self.epoch_times.append(time.time() - start_epoch_time)

    def _create_node_loader(self, data_name):
        data = load_pretrain_single_graph_data(self.configs, data_name)
        node_loader = NeighborLoader(data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                     shuffle=False, num_workers=self.configs.num_workers,
                                     transform=Compose([RootedEgoNets(self.configs.k_hops),
                                                        RenameFromRootedEgoNets()]),
                                     persistent_workers=False)
        return node_loader

    def _create_graph_loader(self, data_name):
        dataset = load_pretrain_multi_graph_data(self.configs, data_name)
        graph_loader = DataLoader(dataset, batch_size=self.configs.batch_size, shuffle=False,
                                  num_workers=self.configs.num_workers, persistent_workers=False)
        return graph_loader