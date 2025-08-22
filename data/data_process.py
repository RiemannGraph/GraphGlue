from torch_geometric.nn.kge import TransE, ComplEx, DistMult, RotatE
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
from torch_geometric.data import Data


class KGNodeInitializer:

    MODEL_CONFIG = {
        'transe': {
            'model_class': TransE,
            'optimizer': lambda params: optim.Adam(params, lr=0.01),
            'kwargs': {}
        },
        'complex': {
            'model_class': ComplEx,
            'optimizer': lambda params: optim.Adagrad(params, lr=0.001, weight_decay=1e-6),
            'kwargs': {}
        },
        'distmult': {
            'model_class': DistMult,
            'optimizer': lambda params: optim.Adam(params, lr=0.0001, weight_decay=1e-6),
            'kwargs': {}
        },
        'rotate': {
            'model_class': RotatE,
            'optimizer': lambda params: optim.Adam(params, lr=1e-3),
            'kwargs': {'margin': 9.0}
        }
    }

    def __init__(self, model_name: str, device: torch.device):
        """

        Args:
            model_name: 'transe', 'complex', 'distmult', 'rotate'
            device: cpu or cuda
        """
        assert model_name in self.MODEL_CONFIG, f"Unsupported model: {model_name}"

        self.model_name = model_name
        self.device = device
        self.model = None
        self.optimizer = None

    def setup_model(self, num_nodes: int, num_relations: int, hidden_channels: int = 50):
        config = self.MODEL_CONFIG[self.model_name]
        self.model = config['model_class'](
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            **config['kwargs']
        ).to(self.device)

        self.optimizer = config['optimizer'](self.model.parameters())

    def create_loader(self, data: Data, batch_size: int, shuffle: bool = True):
        if self.model is None:
            raise RuntimeError("Using setup_model() to initialize the model at first.")

        return self.model.loader(
            head_index=data.edge_index[0],
            rel_type=data.edge_type,
            tail_index=data.edge_index[1],
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = total_examples = 0

        for head_index, rel_type, tail_index in train_loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(head_index, rel_type, tail_index)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()

        return total_loss / total_examples if total_examples > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, data: Data, batch_size: int = 1000, k: int = 10) -> Tuple[float, float, float]:
        self.model.eval()
        rank, mrr, hits = self.model.test(
            head_index=data.edge_index[0],
            rel_type=data.edge_type,
            tail_index=data.edge_index[1],
            batch_size=batch_size,
            k=k,
        )
        return rank, mrr, hits

    def get_node_embeddings(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        return self.model.node_emb.weight.data.clone()

    def get_relation_embeddings(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        return self.model.rel_emb.weight.data.clone()

    def fit(
            self,
            train_data: Data,
            valid_data: Optional[Data] = None,
            test_data: Optional[Data] = None,
            batch_size: int = 1024,
            epochs: int = 500,
            eval_interval: int = 25,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """

        Args:
            train_data:
            valid_data:
            test_data:
            batch_size:
            epochs:
            eval_interval:
            verbose:

        Returns:

        """
        self.setup_model(
            num_nodes=train_data.num_nodes,
            num_relations=train_data.num_edge_types if hasattr(train_data, 'num_edge_types')
            else train_data.edge_type.max().item() + 1
        )

        train_data = train_data.to(self.device)
        valid_data = valid_data.to(self.device)
        test_data = test_data.to(self.device)

        train_loader = self.create_loader(train_data, batch_size)

        results = {'train_loss': [], 'val_metrics': [], 'test_metrics': None}

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            results['train_loss'].append(loss)

            if verbose and epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

            if epoch % eval_interval == 0 and valid_data is not None:
                rank, mrr, hits = self.evaluate(valid_data)
                results['val_metrics'].append((epoch, rank, mrr, hits))

                if verbose:
                    print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
                          f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

        if test_data is not None:
            rank, mrr, hits = self.evaluate(test_data)
            results['test_metrics'] = (rank, mrr, hits)

            if verbose:
                print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
                      f'Test Hits@10: {hits:.4f}')

        results['node_embeddings'] = self.get_node_embeddings()
        results['relation_embeddings'] = self.get_relation_embeddings()

        return results