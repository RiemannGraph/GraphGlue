import os
import time
import torch
from torch_geometric.loader import NeighborLoader, DataLoader, LinkLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from cores.models import RPGraphFM
from utils.checkpoints import save_checkpoint, load_checkpoint, get_latest_checkpoint, cleanup_old_checkpoints, EarlyStopping
from data.data_loader import load_few_shot_multi_graph_data, load_few_shot_single_graph_data
from downstream.tasks import (
    train_node_cls, evaluate_node_cls, train_graph_cls, evaluate_graph_cls,
)
from downstream.adapter import RPGPrompt
from utils.logger import create_logger
from torch_geometric.transforms import RootedEgoNets, Compose
from data.data_process import RenameFromRootedEgoNets


class Trainer:
    def __init__(self, configs, pretrained_model: RPGraphFM, logger=None):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger if logger is not None else create_logger(configs.log_path)

        self.train_loaders = []
        self.val_loaders = []
        self.test_loaders = []
        if configs.task_type == "node_cls":
            self.dataset = load_few_shot_single_graph_data(configs, configs.data_name,
                                                           configs.k_shot, configs.num_trails,
                                                           configs.num_val, configs.num_test)
            data = self.dataset[0]
            for t in range(self.configs.num_trails):
                self.train_loaders.append(NeighborLoader(data, configs.num_neighbors, input_nodes=data.train_mask[:, t], shuffle=True,
                                                   transform=Compose([RootedEgoNets(self.configs.k_hops),
                                                                      RenameFromRootedEgoNets()])
                                                   ))
                self.val_loaders.append(NeighborLoader(data, configs.num_neighbors, input_nodes=data.val_mask[:, t], shuffle=False,
                                                 transform=Compose([RootedEgoNets(self.configs.k_hops),
                                                                    RenameFromRootedEgoNets()])
                                                 ))
                self.test_loaders.append(NeighborLoader(data, configs.num_neighbors, input_nodes=data.test_mask[:, t], shuffle=False,
                                                  transform=Compose([RootedEgoNets(self.configs.k_hops),
                                                                     RenameFromRootedEgoNets()])
                                                  ))
        elif configs.task_type == "graph_cls":
            self.dataset = load_few_shot_multi_graph_data(configs, configs.data_name,
                                                           configs.k_shot, configs.num_trails,
                                                           configs.num_val, configs.num_test)
            #DOTO:
            self.train_loader = DataLoader(self.dataset, batch_size=configs.batch_size, shuffle=True)
        elif configs.task_type == "edge_cls":
            pass

        self.model = RPGPrompt(configs, self.dataset.num_features,
                               pretrained_model, configs.task_type,
                               self.dataset.num_classes
                               ).to(self.device)
        self.model.pretrained_model.frozen()

        self.start_epoch = 0

        self.task_type = configs.task_type

        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)

    def train(self):
        early_stopping = EarlyStopping(
            patience=self.configs.patience,
            mode='max',
            delta=0.001,
            checkpoint_dir=self.configs.checkpoint_dir,
            verbose=True
        )

        optimizer = Adam(
            self.model.parameters(),
            lr=self.configs.lr_task,
            weight_decay=self.configs.task_weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.configs.task_epochs,
            eta_min=self.configs.lr_task * 0.01
        )

        # Resume
        if self.configs.resume_checkpoint:
            latest_ckpt = get_latest_checkpoint(self.configs.checkpoint_dir)
            if latest_ckpt:
                self.start_epoch = load_checkpoint(latest_ckpt, self.model, optimizer, scheduler)
                self.logger.info(f"Resumed from epoch {self.start_epoch}")

        # Train loop
        for
        self.model.train()
        for epoch in range(self.start_epoch, self.configs.task_epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = self._train_epoch(optimizer)
            scheduler.step()
            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f'Epoch {epoch:03d}/{self.configs.task_epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Train ACC: {train_acc * 100:.2f}% | '
                f'Time: {epoch_time:.2f}s | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

            # Evaluation
            if (epoch + 1) % self.configs.eval_interval == 0:
                if self.task_type == 'node_cls':
                    val_loss, val_acc = evaluate_node_cls(self.val_loader, self.model, self.device)
                    self.logger.info(f'Epoch {epoch:03d} | Val Acc: {val_acc:.4f}')
                elif self.task_type == 'graph_cls':
                    val_loss, val_acc = evaluate_graph_cls(self.val_loader, self.model, self.device)
                    self.logger.info(f'Epoch {epoch:03d} | Val Acc: {val_acc:.4f}')
                elif self.task_type == 'edge_cls':
                    pass

                if early_stopping.step(
                        metric=val_acc,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        config=self.configs
                ):
                    break

            # Save
            if (epoch + 1) % self.configs.save_interval == 0 or (epoch + 1) == self.configs.finetune_epochs:
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    config=self.configs.__dict__,
                    filepath=os.path.join(self.configs.checkpoint_dir, f'downstream_epoch_{epoch+1}.pth')
                )

        # Final save
        final_path = os.path.join(self.configs.checkpoint_dir, 'downstream_final.pth')
        torch.save({'state_dict': self.model.state_dict()}, final_path)
        self.logger.info(f"Training finished. Final model saved to {final_path}")

    def _train_epoch(self, optimizer):
        loss = None
        acc = None
        if self.task_type == 'node_cls':
            loss, acc = train_node_cls(self.train_loader, optimizer, self.model, self.device)
        elif self.task_type == 'graph_cls':
            loss, acc = train_graph_cls(self.train_loader, optimizer, self.model, self.device)
        elif self.task_type == 'edge_cls':
            pass
        return loss, acc