import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from cores.models import RPGraphFM
from data import (
    load_few_shot_multi_graph_data,
    load_few_shot_single_graph_data,
    load_few_shot_link_graph_data,
    LinkDataLoader
)
from downstream.adapter import RPGPrompt
from downstream.tasks import train_step, eval_step
from utils.checkpoints import (
    load_checkpoint,
    get_latest_checkpoint,
    EarlyStopping
)
from utils.logger import create_logger


class AdaptTrainer:
    TASK_CONFIGS = {
        'node_cls': {'label_attr': 'y'},
        'graph_cls': {'label_attr': 'y'},
        'link_cls': {'label_attr': 'edge_label'},
    }
    def __init__(self, configs, logger=None):
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger if logger is not None else create_logger(configs.log_path)

        loaders, num_classes, num_features = self.get_loaders(configs)
        self.train_loaders = loaders[0]
        self.val_loaders = loaders[1]
        self.test_loaders = loaders[2]

        pretrained_model = RPGraphFM(configs)
        load_checkpoint(configs.pretrained_checkpoint, pretrained_model, map_location='cuda')
        self.model = RPGPrompt(configs, num_features,
                               pretrained_model, configs.task_type,
                               num_classes
                               ).to(self.device)

        self.start_epoch = 0

        self.task_type = configs.task_type

        os.makedirs(self.configs.checkpoint_dir, exist_ok=True)

    def train(self):
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
        total_acc = []
        for trial in range(self.configs.num_trials):
            early_stopping = EarlyStopping(
                patience=self.configs.patience,
                mode='max',
                delta=0.001,
                checkpoint_dir=self.configs.checkpoint_dir,
                verbose=True
            )
            self.model.train()
            for epoch in range(self.start_epoch, self.configs.task_epochs):
                epoch_start_time = time.time()
                train_loss, train_acc = self._train_epoch(optimizer, trial)
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
                    val_loss, val_acc = eval_step(self.val_loaders[trial], self.model, self.device,
                                           **AdaptTrainer.TASK_CONFIGS[self.task_type])
                    self.logger.info(f'Epoch {epoch:03d} | Val Acc: {val_acc * 100:.2f}%')

                    if early_stopping.step(
                            metric=val_acc,
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            config=self.configs
                    ):
                        break

            # Final save
            final_path = os.path.join(self.configs.checkpoint_dir, f'downstream_final_{trial}.pth')
            torch.save({'state_dict': self.model.state_dict()}, final_path)
            self.logger.info(f"Trial {trial} | Training finished. Final model saved to {final_path}")

            test_loss, test_acc = eval_step(self.test_loaders[trial], self.model, self.device,
                                            **AdaptTrainer.TASK_CONFIGS[self.task_type])
            self.logger.info(f'Trial {trial:03d} | Test Acc: {test_acc * 100:.2f}%')
            total_acc.append(test_acc)
        self.logger.info(f'Final Test Acc: {np.mean(total_acc) * 100:.2f} \u00B1 {np.std(total_acc) * 100:.2f} %')

    def _train_epoch(self, optimizer, trial):
        loss = None
        acc = None
        loss, acc = train_step(self.train_loaders[trial], optimizer, self.model, self.device,
                               **AdaptTrainer.TASK_CONFIGS[self.task_type])
        return loss, acc

    def get_loaders(self, configs):
        num_classes = None
        num_features = None
        train_loaders = []
        val_loaders = []
        test_loaders = []
        if configs.task_type == "node_cls":
            dataset, train_mask, val_mask, test_mask = load_few_shot_single_graph_data(configs, configs.data_name,
                                                                                      configs.k_shot,
                                                                                      configs.num_trials,
                                                                                      configs.num_val)
            num_classes = dataset.num_classes
            num_features = dataset.num_features
            for t in range(configs.num_trials):
                train_loaders.append(DataLoader(dataset[train_mask[:, t]],
                                                batch_size=configs.batch_size, shuffle=True,
                                                exclude_keys=["original_node_ids", "center_node_idx", "edge_attr"]))
                val_loaders.append(DataLoader(dataset[val_mask[:, t]],
                                              batch_size=configs.batch_size, shuffle=False,
                                              exclude_keys=["original_node_ids", "center_node_idx", "edge_attr"]))
                test_loaders.append(DataLoader(dataset[test_mask[:, t]],
                                               batch_size=configs.batch_size, shuffle=False,
                                               exclude_keys=["original_node_ids", "center_node_idx", "edge_attr"]))
        elif configs.task_type == "graph_cls":
            dataset, train_mask, val_mask, test_mask = load_few_shot_multi_graph_data(configs, configs.data_name,
                                                           configs.k_shot, configs.num_trials,
                                                           configs.num_val)
            num_classes = dataset.num_classes
            num_features = dataset.num_features
            for t in range(configs.num_trials):
                train_loaders.append(DataLoader(dataset[train_mask[:, t]], batch_size=configs.batch_size, shuffle=True))
                val_loaders.append(DataLoader(dataset[val_mask[:, t]], batch_size=configs.batch_size, shuffle=False))
                test_loaders.append(DataLoader(dataset[test_mask[:, t]], batch_size=configs.batch_size, shuffle=False))

        elif configs.task_type == "link_cls":
            data, train_sets, val_sets, test_sets = load_few_shot_link_graph_data(configs, configs.data_name,
                                                                 configs.k_shot, configs.num_trials,
                                                                 configs.num_val)
            num_classes = len(data.edge_type.unique())
            num_features = data.x.shape[-1]
            for t in range(configs.num_trials):
                train_loaders.append(LinkDataLoader(train_sets[t], batch_size=configs.batch_size, shuffle=True))
                val_loaders.append(LinkDataLoader(val_sets[t], batch_size=configs.batch_size, shuffle=False))
                test_loaders.append(LinkDataLoader(test_sets[t], batch_size=configs.batch_size, shuffle=False))
        else:
            raise NotImplementedError
        return (train_loaders, val_loaders, test_loaders), num_classes, num_features