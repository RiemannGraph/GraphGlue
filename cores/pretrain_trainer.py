import torch
from cores.models import RPGraphFM
from data.data_loader import load_pretrain_single_graph_data, load_pretrain_multi_graph_data
from torch_geometric.loader import DataLoader, NeighborLoader
from utils.logger import create_logger
from utils.checkpoints import save_checkpoint, load_checkpoint, get_latest_checkpoint, cleanup_old_checkpoints
import os
from typing import List
from torch_geometric.transforms import RootedEgoNets, Compose
from data.data_process import RenameFromRootedEgoNets
import time
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

    def _train_epoch(self, optimizer, epoch):
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        all_graph_prototypes = []
        all_graph_tan_prototypes = []

        # Handling node-level graph datasets
        num_node_level_datasets = len(self.pretrain_single_graph_data)
        if num_node_level_datasets > 0:
            for data_idx, data_name in enumerate(self.pretrain_single_graph_data):
                self.logger.info(f'Processing node loader {data_idx + 1}/{num_node_level_datasets}')

                graph_z_list = []
                graph_z_tan_list = []

                node_loader = self._create_node_loader(data_name)
                for batch_idx, data in enumerate(node_loader):
                    optimizer.zero_grad()
                    data = data.to(self.device)
                    z, z_tan = self.model(data, data.batch_graph_nums)
                    intra_loss = self.model.loss(z, z_tan, data.origin_edge_index, node_loader.batch_size)
                    intra_loss.backward()
                    # if self.configs.max_grad_norm > 0:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.max_grad_norm)
                    optimizer.step()

                    graph_z_list.append(z[:node_loader.batch_size].detach().cpu())
                    graph_z_tan_list.append(z_tan[:node_loader.batch_size].detach().cpu())

                    total_loss += intra_loss.item()
                    total_batches += 1

                    if batch_idx % self.configs.log_interval == 0:
                        self.logger.info(
                            f'Epoch {epoch} | Node Loader {data_idx} | '
                            f'Batch {batch_idx} | Loss: {intra_loss.item():.6f}'
                        )

                # compute current graph prototypes
                if graph_z_list:
                    g_rep = torch.cat(graph_z_list, dim=0).mean(dim=0, keepdim=True)  # [1, dim]
                    g_rep_tan = torch.cat(graph_z_tan_list, dim=0).mean(dim=0, keepdim=True)
                    all_graph_prototypes.append(g_rep)
                    all_graph_tan_prototypes.append(g_rep_tan)

                del node_loader, graph_z_list, graph_z_tan_list
                gc.collect()

        # Handling graph-level graph datasets
        num_graph_level_datasets = len(self.pretrain_multi_graph_data)
        if num_graph_level_datasets > 0:
            for data_idx, data_name in enumerate(self.pretrain_multi_graph_data):
                self.logger.info(f'Processing graph loader {data_idx + 1}/{num_graph_level_datasets}')
                graph_loader = self._create_graph_loader(data_name)
                for batch_idx, data in enumerate(graph_loader):
                    optimizer.zero_grad()
                    data = data.to(self.device)
                    z, z_tan = self.model(data, data.batch_size)
                    edge_index, _ = self.model.knn_graph(z, self.configs.knn)
                    intra_loss = self.model.loss(z, z_tan, edge_index)
                    intra_loss.backward()
                    # if self.configs.max_grad_norm > 0:
                    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.max_grad_norm)
                    optimizer.step()

                    all_graph_prototypes.append(z.detach().cpu())
                    all_graph_tan_prototypes.append(z_tan.detach().cpu())

                    total_loss += intra_loss.item()
                    total_batches += 1

                del graph_loader
                gc.collect()

        # Handling cross datasets
        if len(all_graph_prototypes) > 0:
            optimizer.zero_grad()
            all_graph_embeds = torch.cat(all_graph_prototypes, dim=0).to(self.device)  # [N_total, dim]
            all_graph_tans = torch.cat(all_graph_tan_prototypes, dim=0).to(self.device)
            edge_index = self.model.knn_graph(all_graph_embeds, self.configs.knn)
            inter_loss = self.model.loss(all_graph_embeds, all_graph_tans, edge_index)
            inter_loss.backward()
            # if self.configs.max_grad_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.max_grad_norm)
            optimizer.step()

            total_loss += inter_loss.item()
            total_batches += 1
            self.logger.info(f'Epoch {epoch} | Inter Loss: {inter_loss.item():.6f}')
            del all_graph_embeds, all_graph_tans
        else:
            self.logger.warning("Not enough graph prototypes for inter-loss.")
        return total_loss / total_batches

    @torch.no_grad()
    def register_from_loaders(self):
        if self.final_model_path:
            load_checkpoint(self.final_model_path, self.model, map_location='cuda' if torch.cuda.is_available() else 'cpu')
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