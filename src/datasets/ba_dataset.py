import os
import json
import hashlib
from typing import Callable, Dict, List, Optional, Union
 
import numpy as np
import torch
import networkx as nx
 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets.graph_generator import BAGraph
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
 
SUPPORTED_METRICS = [
    "clustering_coefficient",
    "diameter",
    "avg_degree",
    "density",
    "transitivity",
]
 
 
def _compute_metric(G: nx.Graph, metric: str) -> float:
    """Return the raw (continuous) value of *metric* for graph *G*."""
    if metric == "clustering_coefficient":
        return nx.average_clustering(G)
    elif metric == "diameter":
        if not nx.is_connected(G):
            # Use the largest connected component's diameter
            largest_cc = max(nx.connected_components(G), key=len)
            return nx.diameter(G.subgraph(largest_cc))
        return float(nx.diameter(G))
    elif metric == "avg_degree":
        degrees = [d for _, d in G.degree()]
        return float(np.mean(degrees))
    elif metric == "density":
        return nx.density(G)
    elif metric == "transitivity":
        return nx.transitivity(G)
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported: {SUPPORTED_METRICS}"
        )
 
 
def _build_nx(data: Data) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.numpy()
    for i in range(ei.shape[1]):
        G.add_edge(int(ei[0, i]), int(ei[1, i]))
    return G
 
class BAGraphDataset(InMemoryDataset):
    """
    Parameters
    ----------
    root : str
        Root directory.  Processed data is cached under ``root/processed/``.
    num_graphs : int
        How many BA graphs to generate.
    num_nodes : int
        Number of nodes per graph (passed to BAGraph).
    num_edges : int
        Number of edges to attach per new node (the *m* parameter of BA).
    metrics : list[str]
        Which structural metrics to include.  Defaults to
        ``["clustering_coefficient", "diameter"]``.
    num_classes : int | dict[str, int]
        Number of bins for each metric.  Pass a single int to use the same
        number for every metric, or a dict to set per-metric values.
    transform : callable, optional
        Transform applied to each graph at access time.
    pre_transform : callable, optional
        Transform applied once before saving.
    seed : int, optional
        Random seed for reproducibility.
 
    Output ``data.y``
    -----------------
    Shape ``[1, len(metrics)]`` of dtype ``torch.long`` – bin index for each
    metric.  This mirrors QM9's ``data.y`` layout (shape ``[1, 19]``).
    Raw continuous values are stored in ``data.y_raw`` (shape ``[1, len(metrics)]``,
    dtype ``torch.float``).
 
    Bin edges are available as ``dataset.bin_edges``:
    a dict mapping metric name → 1-D numpy array of edges.
    """
 
    def __init__(
        self,
        root: str,
        num_graphs: int = 1000,
        num_nodes: int = 100,
        num_edges: int = 2,
        metrics: Optional[List[str]] = None,
        num_classes: Union[int, Dict[str, int]] = 5,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        # Resolve metrics
        if metrics is None:
            metrics = ["clustering_coefficient", "diameter"]
        for m in metrics:
            if m not in SUPPORTED_METRICS:
                raise ValueError(
                    f"Unsupported metric '{m}'. Choose from {SUPPORTED_METRICS}."
                )
        self.metric_names: List[str] = metrics
 
        # Resolve num_classes per metric
        if isinstance(num_classes, int):
            self._num_classes: Dict[str, int] = {m: num_classes for m in metrics}
        else:
            for m in metrics:
                if m not in num_classes:
                    raise ValueError(f"num_classes missing entry for metric '{m}'.")
            self._num_classes = dict(num_classes)
 
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.seed = seed
 
        # Bin edges computed after processing; loaded from cache if available
        self.bin_edges: Dict[str, np.ndarray] = {}
 
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
 
        # Load bin edges from sidecar JSON
        with open(self._bin_edges_path(), "r") as f:
            raw = json.load(f)
        self.bin_edges = {k: np.array(v) for k, v in raw.items()}
 
    @property
    def raw_file_names(self) -> List[str]:
        return []          # Nothing to download
 
    @property
    def processed_file_names(self) -> List[str]:
        return [f"ba_graphs_{self._config_hash()}.pt"]
 
    def download(self):
        pass
 
    def _config_hash(self) -> str:
        """Short hash encoding all hyper-parameters that affect the data."""
        cfg = {
            "num_graphs": self.num_graphs,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "metrics": self.metric_names,
            "num_classes": self._num_classes,
            "seed": self.seed,
        }
        h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
        return h
 
    def _bin_edges_path(self) -> str:
        name = f"bin_edges_{self._config_hash()}.json"
        return os.path.join(self.processed_dir, name)
 
    def process(self):
        rng = np.random.default_rng(self.seed)
        generator = BAGraph(num_nodes=self.num_nodes, num_edges=self.num_edges)
 
        data_list: List[Data] = []
        raw_values: Dict[str, List[float]] = {m: [] for m in self.metric_names}
 
        print(f"Generating {self.num_graphs} BA graphs …")
        for i in range(self.num_graphs):
            # Use a different seed per graph but deterministically
            torch.manual_seed(int(rng.integers(1 << 31)))
            data = generator()
 
            G = _build_nx(data)
 
            vals = []
            for m in self.metric_names:
                v = _compute_metric(G, m)
                raw_values[m].append(v)
                vals.append(v)
 
            data.y_raw = torch.tensor([vals], dtype=torch.float)   # [1, M]
 
            if self.pre_transform is not None:
                data = self.pre_transform(data)
 
            data_list.append(data)
 
        # ---- Compute bin edges from the full distribution ----
        bin_edges: Dict[str, np.ndarray] = {}
        for m in self.metric_names:
            arr = np.array(raw_values[m])
            k = self._num_classes[m]
            # Use quantile-based edges so each bin is equally populated
            quantiles = np.linspace(0, 100, k + 1)
            edges = np.percentile(arr, quantiles)
            # Make the last edge slightly larger to include the maximum
            edges[-1] += 1e-9
            bin_edges[m] = edges
 
        # ---- Assign bin labels ----
        for i, data in enumerate(data_list):
            labels = []
            for j, m in enumerate(self.metric_names):
                raw_val = data.y_raw[0, j].item()
                bin_idx = int(np.searchsorted(bin_edges[m][1:], raw_val))
                bin_idx = min(bin_idx, self._num_classes[m] - 1)
                labels.append(bin_idx)
            data.y = torch.tensor([labels], dtype=torch.long)      # [1, M]
 
        # ---- Persist ----
        self.save(data_list, self.processed_paths[0])
 
        with open(self._bin_edges_path(), "w") as f:
            json.dump({k: v.tolist() for k, v in bin_edges.items()}, f, indent=2)
 
        print("Done.")
 
    def num_classes_for(self, metric: str) -> int:
        """Return the number of bins for *metric*."""
        return self._num_classes[metric]
 
    def decode_label(self, y: torch.Tensor) -> Dict[str, int]:
        """
        Given a label tensor of shape ``[1, M]`` or ``[M]``, return a dict
        mapping metric name to bin index.
        """
        y = y.view(-1)
        return {m: int(y[i].item()) for i, m in enumerate(self.metric_names)}
 
    def decode_bin_range(self, metric: str, bin_idx: int):
        """
        Return the (low, high) raw-value range for a given bin.
        """
        edges = self.bin_edges[metric]
        return (float(edges[bin_idx]), float(edges[bin_idx + 1]))
 
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_graphs={len(self)}, "
            f"num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"metrics={self.metric_names}, "
            f"num_classes={self._num_classes})"
        )
        
class BAGraphDataModule:
    """
    Thin Lightning-style datamodule wrapping BAGraphDataset.
    Mirrors the interface expected by the training loop and dataset_infos.
    """
 
    def __init__(
        self,
        cfg: DictConfig,
        num_graphs: int = 10000,
        num_nodes: int = 50,
        num_edges: int = 2,
        metrics=None,
        num_classes=5,
        seed: int = 42,
    ):
        self.cfg = cfg
        batch_size  = cfg["train"]["batch_size"]
        num_workers = cfg.get("num_workers", 4)
        root        = cfg.dataset.get("root", "data/ba_graph")
 
        # Split sizes
        n_train = int(num_graphs * 0.8)
        n_val   = int(num_graphs * 0.1)
        n_test  = num_graphs - n_train - n_val
 
        # Generate the full dataset once; the cache key encodes all params
        full = BAGraphDataset(
            root=root,
            num_graphs=num_graphs,
            num_nodes=num_nodes,
            num_edges=num_edges,
            metrics=metrics,
            num_classes=num_classes,
            seed=seed,
        )
 
        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                full,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(seed),
            )
        )
 
        # Expose the underlying BAGraphDataset so main.py can reach .bin_edges
        # random_split wraps the dataset; unwrap to the original
        self._full_dataset = full
 
        # Expose the train split as a BAGraphDataset proxy for bin_edges access
        # (main.py does datamodule.train_dataset.bin_edges[...])
        self.train_dataset.bin_edges   = full.bin_edges
        self.train_dataset.metric_names = full.metric_names
 
        self._batch_size  = batch_size
        self._num_workers = num_workers
 
    # ---- Dataloaders (same names as QM9DataModule) ----
 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
        )
 
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
 
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
 
    # ---- Properties inspected by DatasetInfos ----
 
    @property
    def num_graphs(self):
        return len(self._full_dataset)
 
    @property
    def metric_names(self):
        return self._full_dataset.metric_names
 
    @property
    def bin_edges(self):
        return self._full_dataset.bin_edges

class BAGraphDatasetInfos:
    """
    Mirrors QM9infos / SpectreDatasetInfos.
    Provides input_dims, output_dims, and graph statistics to the model.
    """
 
    def __init__(self, datamodule: BAGraphDataModule, cfg: DictConfig):
        self.datamodule  = datamodule
        self.cfg         = cfg
        self.name        = "ba_graph"
        self.metric_names = datamodule.metric_names
 
        # Number of conditioning targets == number of metrics
        self.num_targets = len(self.metric_names)
 
        # Collect basic graph statistics from the train split
        self._compute_statistics(datamodule)
 
    def _compute_statistics(self, datamodule: BAGraphDataModule):
        """Compute max node count, node/edge type distributions."""
        max_n, node_counts = 0, []
        for batch in datamodule.train_dataloader():
            n = batch.num_nodes if hasattr(batch, "num_nodes") else batch.x.shape[0]
            max_n = max(max_n, int(n))
            node_counts.append(n)
 
        self.max_n_nodes  = max_n
        self.n_nodes      = torch.tensor(node_counts, dtype=torch.float)
 
        # BA graphs are simple unweighted: 1 node type, 1 edge type (+ no-edge)
        self.node_types   = torch.tensor([1.0])          # single node feature
        self.edge_types   = torch.tensor([0.5, 0.5])     # no-edge / edge
 
    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        """
        Populate input_dims and output_dims exactly as QM9infos does,
        so the model can infer channel sizes.
 
        BA graphs have:
          X : node features  — 1-hot node type (1 class) + extra_features
          E : edge features  — 1-hot edge type (2 classes: absent/present) + extra_features
          y : graph label    — num_targets (one per conditioning metric)
        """
        # Probe extra-feature dimensions via a dummy batch
        sample = next(iter(datamodule.train_dataloader()))
 
        ex_dims = extra_features.get_dims() if hasattr(extra_features, 'get_dims') else {'X': 0, 'E': 0, 'y': 0}
        dom_dims = domain_features.get_dims() if hasattr(domain_features, 'get_dims') else {'X': 0, 'E': 0, 'y': 0}
 
        # Node feature dim: base (node type one-hot) + extras
        base_X = len(self.node_types)          # 1
        base_E = len(self.edge_types)          # 2
        base_y = self.num_targets              # number of conditioning metrics
 
        self.input_dims = {
            'X': base_X + ex_dims.get('X', 0) + dom_dims.get('X', 0),
            'E': base_E + ex_dims.get('E', 0) + dom_dims.get('E', 0),
            'y': base_y + ex_dims.get('y', 0) + dom_dims.get('y', 0),
        }
 
        # Output dims are the generative targets (same as input for graph gen)
        self.output_dims = {
            'X': base_X,
            'E': base_E,
            'y': base_y,
        }