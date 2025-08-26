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


class Pretrainer:
    def __init__(self, configs, logger=None):
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

        node_loaders, graph_loaders = self._get_loaders()

        for epoch in range(self.start_epoch, self.configs.pretrain_epochs):
            epoch_start_time = time.time()

            train_loss = self._train_epoch(optimizer, node_loaders, graph_loaders, epoch)

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
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'config': self.configs.__dict__,
                }, final_model_path)
                self.logger.info(f'Saved final model: {final_model_path}')

    def _train_epoch(self, optimizer, node_loaders: List[NeighborLoader], graph_loaders: List[DataLoader], epoch):
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        all_graph_embeds = []
        all_graph_tans = []

        for loader_idx, node_loader in enumerate(node_loaders):
            self.logger.info(f'Processing node loader {loader_idx + 1}/{len(node_loaders)}')
            g_rep = torch.tensor([])
            g_rep_tan = torch.tensor([])
            for batch_idx, data in enumerate(node_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                z, z_tan = self.model(data, data.batch_graph_nums)
                intra_loss = self.model.loss(z, z_tan, data.origin_edge_index, data.batch_size)
                intra_loss.backward()
                # if self.configs.max_grad_norm > 0:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.max_grad_norm)
                optimizer.step()

                g_rep = torch.concat([g_rep, z.cpu()], dim=0)
                g_rep_tan = torch.concat([g_rep_tan, z_tan.cpu()], dim=0)

                total_loss += intra_loss.item()
                total_batches += 1
                if batch_idx % self.configs.log_interval == 0:
                    self.logger.info(
                        f'Epoch {epoch} | Node Loader {loader_idx} | '
                        f'Batch {batch_idx} | Loss: {intra_loss.item():.6f}'
                    )

            all_graph_embeds.append(g_rep.mean(dim=0, keepdim=True))
            all_graph_tans.append(g_rep_tan.mean(dim=0, keepdim=True))

        for loader_idx, graph_loader in enumerate(graph_loaders):
            self.logger.info(f'Processing graph loader {loader_idx + 1}/{len(graph_loaders)}')
            for batch_idx, data in enumerate(graph_loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                z, z_tan = self.model(data, data.batch_size)
                edge_index = self.model.knn_graph(z, self.configs.knn)
                intra_loss = self.model.loss(z, z_tan, edge_index)
                intra_loss.backward()
                # if self.configs.max_grad_norm > 0:
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.max_grad_norm)
                optimizer.step()

                all_graph_embeds.append(z.cpu())
                all_graph_tans.append(z_tan.cpu())
                total_loss += intra_loss.item()
                total_batches += 1

        optimizer.zero_grad()
        all_graph_embeds = torch.concat(all_graph_embeds, dim=0).to(self.device)
        all_graph_tans = torch.concat(all_graph_tans, dim=0).to(self.device)
        edge_index = self.model.knn_graph(all_graph_embeds, self.configs.knn)
        inter_loss = self.model.loss(all_graph_embeds, all_graph_tans, edge_index)
        inter_loss.backward()
        # if self.configs.max_grad_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.max_grad_norm)
        optimizer.step()
        total_loss += inter_loss.item()
        total_batches += 1
        self.logger.info(f'Epoch {epoch} | Inter Loss: {inter_loss.item():.6f}')

        return total_loss / total_batches

    def _get_loaders(self):
        node_loaders = []
        graph_loaders = []
        for data_name in self.pretrain_single_graph_data:
            data = load_pretrain_single_graph_data(self.configs, data_name)
            node_loader = NeighborLoader(data, batch_size=self.configs.batch_size,
                                         num_neighbors=self.configs.num_neighbors,
                                         shuffle=False, num_workers=8, disjoint=False,
                                         transform=Compose([RootedEgoNets(self.configs.k_hops),
                                                            RenameFromRootedEgoNets()]))
            node_loaders.append(node_loader)
        for data_name in self.pretrain_multi_graph_data:
            dataset = load_pretrain_multi_graph_data(self.configs, data_name)
            graph_loader = DataLoader(dataset, batch_size=self.configs.batch_size, shuffle=False, num_workers=8)
            graph_loaders.append(graph_loader)
        return node_loaders, graph_loaders