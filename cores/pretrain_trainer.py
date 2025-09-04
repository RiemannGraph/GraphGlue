import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.transforms import RootedEgoNets, Compose
from cores.models import RPGraphFM
from data.data_loader import load_pretrain_single_graph_data, load_pretrain_multi_graph_data
from utils.logger import create_logger
from utils.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    cleanup_old_checkpoints)
import os
from data.data_transform import RenameFromRootedEgoNets
from utils.tools import format_time
import time
import gc
import warnings

warnings.filterwarnings("ignore")


class Pretrainer:
    def __init__(self, configs, logger=None):
        self.final_model_path = None
        assert len(configs.num_neighbors) == configs.k_hops, "number of neighbor hops are not match!"
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

        self.tmp_checkpoint_dir = os.path.join(self.configs.checkpoint_dir, 'tmp')
        os.makedirs(self.tmp_checkpoint_dir, exist_ok=True)

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
        resume_from = None
        if self.configs.resume_checkpoint and self.configs.resume_temp_checkpoint:
            raise ValueError("Conflicting resume settings...")

        if self.configs.resume_checkpoint:
            latest_check_path = get_latest_checkpoint(self.configs.checkpoint_dir)
            if latest_check_path:
                self.start_epoch = load_checkpoint(latest_check_path, self.model, optimizer, scheduler)
                self.logger.info(f"Resumed from main checkpoint at epoch {self.start_epoch}")
            else:
                self.start_epoch = 0

        elif self.configs.resume_temp_checkpoint:
            latest_temp_path = get_latest_checkpoint(self.tmp_checkpoint_dir)
            if latest_temp_path:
                self.start_epoch, temp_config = load_checkpoint(
                    filepath=latest_temp_path,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    return_config=True
                )
                resume_from = temp_config.get('resume_from')
                self.logger.info(f"Resumed from temp checkpoint at epoch {self.start_epoch}, step: {resume_from}")
            else:
                self.start_epoch = 0
        else:
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.configs.pretrain_epochs):
            epoch_start_time = time.time()

            train_loss = self._train_epoch(optimizer, scheduler, epoch, resume_from=resume_from)

            if epoch == self.start_epoch:
                resume_from = None
                self.logger.info("Resume flag cleared after first epoch.")

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

            cleanup_old_checkpoints(self.tmp_checkpoint_dir, keep_last=3)

    def _train_epoch(self, optimizer, scheduler, epoch, resume_from=None):
        if self.start_time is None:
            self.start_time = time.time()
        start_epoch_time = time.time()

        self.model.train()
        total_loss = 0.0
        total_batches = 0

        # Phase 1: Intra Training
        loss_stats = self._train_node_level(optimizer, scheduler, epoch, resume_from)
        total_loss += loss_stats['loss']
        total_batches += loss_stats['batches']

        loss_stats = self._train_graph_level(optimizer, scheduler, epoch, resume_from)
        total_loss += loss_stats['loss']
        total_batches += loss_stats['batches']

        # Log
        self._log_epoch_summary(epoch, start_epoch_time)
        self._update_epoch_time(epoch, start_epoch_time)

        return total_loss / total_batches

    def _train_node_level(self, optimizer, scheduler, epoch, resume_from=None):
        return self._train_one_type(
            data_names=self.pretrain_single_graph_data,
            loader_creator=self._create_node_loader,
            type_str='node',
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            resume_from=resume_from
        )

    def _train_graph_level(self, optimizer, scheduler, epoch, resume_from=None):
        return self._train_one_type(
            data_names=self.pretrain_multi_graph_data,
            loader_creator=self._create_graph_loader,
            type_str='graph',
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            resume_from=resume_from
        )

    def _train_one_type(self, data_names, loader_creator, type_str, optimizer, scheduler, epoch, resume_from=None):
        """
        type_str: 'node' or 'graph'
        loader_creator: _create_node_loader or _create_graph_loader
        """
        total_loss = 0.0
        total_batches = 0

        num_datasets = len(data_names)
        if num_datasets == 0:
            return {'loss': 0.0, 'batches': 0}

        start_idx = 0
        if resume_from:
            if resume_from['step_type'] == type_str:
                try:
                    start_idx = data_names.index(resume_from['last_data_name']) + 1
                except ValueError:
                    start_idx = 0
                if start_idx >= num_datasets:
                    return {'loss': 0.0, 'batches': 0}
            elif resume_from['step_type'] == 'node' and type_str == 'graph':
                start_idx = 0
            elif resume_from['step_type'] == 'graph' and type_str == 'node':
                self.logger.info("Already passed node-level stage. Skipping node-level training.")
                return {'loss': 0.0, 'batches': 0}
            else:
                self.logger.warning(f"Unknown resume state: step_type={resume_from['step_type']}, current={type_str}")
                return {'loss': 0.0, 'batches': 0}

        if start_idx >= num_datasets:
            return {'loss': 0.0, 'batches': 0}

        if start_idx > 0:
            self.logger.info(f"Resuming {type_str}-level from dataset {start_idx}: {data_names[start_idx]}")

        for data_idx, data_name in enumerate(data_names):
            if data_idx < start_idx:
                continue

            self.logger.info(f'Processing {type_str} loader {data_idx + 1}/{num_datasets}')
            loader = loader_creator(data_name)
            dataset_len = len(loader)
            start_loader_time = time.time()

            for batch_idx, data in enumerate(loader):
                optimizer.zero_grad()
                data = data.to(self.device)

                # -------------Forward ------------
                if type_str == 'node':
                    batch_size_arg = data.batch_graph_nums
                else:  # 'graph'
                    batch_size_arg = data.batch_size

                z, z_tan = self.model(data, batch_size_arg)

                # ------------ Edge Index -----------
                if type_str == 'node':
                    edge_index = data.origin_edge_index
                    loss_batch_size = loader.batch_size
                else:  # 'graph'
                    edge_index, _ = self.model.knn_graph(z, self.configs.knn)
                    loss_batch_size = None

                # --------------- Loss--------------
                intra_loss = self.model.loss(z, z_tan, edge_index, batch_size=loss_batch_size)

                if epoch >= self.configs.warmup_epochs:
                    proto_loss = self.model.prototype_loss(data_name, z)
                    intra_loss += proto_loss

                intra_loss.backward()
                optimizer.step()

                # ------------- Update Prototype-------------
                if loss_batch_size and type_str == 'node':
                    z_mean = z[:loss_batch_size].mean(dim=0).detach().clone()
                    z_tan_mean = z_tan[:loss_batch_size].mean(dim=0).detach().clone()
                else:
                    z_mean = z.mean(dim=0).detach().clone()
                    z_tan_mean = z_tan.mean(dim=0).detach().clone()

                self.model.update_prototype(data_name, z_mean, z_tan_mean)

                del z_mean, z_tan_mean, data, z, z_tan

                total_loss += intra_loss.item()
                total_batches += 1

                # -------------- Logging ----------------
                if batch_idx % self.configs.log_interval == 0:
                    self._log_progress(
                        epoch=epoch,
                        loader_type=type_str,
                        loader_idx=data_idx + 1,
                        batch_idx=batch_idx,
                        dataset_len=dataset_len,
                        loss=intra_loss.item(),
                        start_loader_time=start_loader_time,
                        batches_done=batch_idx + 1
                    )

            # -------------Checkpoint -----------------
            self._save_temp_checkpoint(
                data_name=data_name,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step_type=type_str,
                data_idx=data_idx
            )
            del loader
            torch.cuda.empty_cache()

        return {'loss': total_loss, 'batches': total_batches}

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

    def _save_temp_checkpoint(self, data_name, optimizer, scheduler, epoch, step_type="node", data_idx=0):
        temp_model_path = os.path.join(
            self.tmp_checkpoint_dir,
            f'temp_pretrain_epoch_{epoch}.pth'
        )
        temp_config = self.configs.__dict__.copy()
        temp_config['resume_from'] = {
            'epoch': epoch,
            'step_type': step_type,
            'last_data_name': data_name,
            'last_data_idx': data_idx,
            'next_step_type': step_type
        }
        save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=temp_config,
            filepath=temp_model_path
        )
        self.logger.info(f"Saved temporary checkpoint: {temp_model_path}")