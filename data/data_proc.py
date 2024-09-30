import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull, CitationFull
from torch_geometric.datasets import Reddit, Flickr
from torch_geometric.datasets import LRGBDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Sampler, Subset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import os
import os.path as osp

np.random.seed(42)
torch.manual_seed(42)

def load_data(cfg):
    dataset = cfg.data.dataset
    path = cfg.data.data_root
    dataset = dataset.lower()

    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']

    except FileNotFoundError:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dataset)
        elif dataset == 'cora_ml':
            dataset = CitationFull(path, dataset)
        elif dataset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dataset)
        elif dataset == 'actor':
            dataset = Actor(path)
        elif dataset in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(path, dataset)
        elif dataset in ['computers', 'photo']:
            dataset = Amazon(path, dataset)
        elif dataset in ['cs', 'physics']:
            dataset = Coauthor(path, dataset)

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    data.edge_index = to_undirected(data.edge_index, num_nodes =data.x.shape[0])

    return data, num_features, num_classes

def load_data_pyg_large(cfg):
    dataset = cfg.data.dataset
    path = cfg.data.data_root
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']

    except FileNotFoundError:
        if dataset in ['reddit']:
            dataset = Reddit(path)
        elif dataset in ['flickr']:
            dataset = Flickr(path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        data = dataset[0]
        num_features = data.x.size(1)
        num_classes = dataset.num_classes

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    data.edge_index = to_undirected(data.edge_index, num_nodes =data.x.shape[0])

    return data, num_features, num_classes

def load_data_ogb(cfg):
    dataset = cfg.data.dataset
    path = cfg.data.data_root
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']

    except FileNotFoundError:
        if dataset in ['ogbn-arxiv', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=dataset, transform=T.ToSparseTensor())
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")


        data = dataset[0]
        num_features = data.x.shape[1]
        num_classes = dataset.num_classes

        # Reshape y
        data.y = data.y.squeeze()

        # Generate train, val, test masks
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    # Make the adjacency matrix to symmetric
    data.adj_t = data.adj_t.to_symmetric()
    # Convert SparseTensor to edge_index
    edge_index = data.adj_t.coo()  # Get COO format
    data.edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    # Ensure the edge_index is in LongTensor format
    data.edge_index = data.edge_index.long()

    return data, num_features, num_classes

def generate_split(data, num_classes, seed=2021, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask

def custom_collate_fn(batch):
    return Batch.from_data_list(batch)

# We will apply channel-wise normalization to data.x
def channel_wise_normalization(data, mean, std):
    data.x = (data.x - mean) / std
    return data


### Define a function to load graph data
def load_data_lrgb(cfg):
    dataset = cfg.data.dataset
    path = cfg.data.data_root
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset_train = torch.load(osp.join(path, 'dataset_train.pt'))
        dataset_val = torch.load(osp.join(path, 'dataset_val.pt'))
        dataset_test = torch.load(osp.join(path, 'dataset_test.pt'))

        train_loader = dataset_train['loader']
        val_loader = dataset_val['loader']
        test_loader = dataset_test['loader']

        num_features = dataset_train['num_features']
        num_classes = dataset_train['num_classes']

    except FileNotFoundError:
        if dataset in ['pascalvoc-sp']:
            dataset_train = LRGBDataset(path, dataset, split = 'train')
            dataset_val = LRGBDataset(path, dataset, split='val')
            dataset_test = LRGBDataset(path, dataset, split='test')
        elif dataset in ['coco-sp']:
            dataset_train = LRGBDataset(path, dataset, split='train')
            dataset_val = LRGBDataset(path, dataset, split='val')
            dataset_test = LRGBDataset(path, dataset, split='test')
            dataset_train = dataset_train[:11329]
            dataset_val = dataset_val[:500]
            dataset_test = dataset_test[:500]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Find num_features and num_classes
        sample_data = dataset_train[0]
        num_features = sample_data.x.size(1)
        num_classes = dataset_train.num_classes

        ### Preprocess training data
        # Compute mean and std for channel-wise normalization
        x_all = torch.cat([data.x for data in dataset_train], dim=0)
        mean = x_all.mean(dim=0)
        std = x_all.std(dim=0)

        # Apply normalization to the training data
        dataset_train = [channel_wise_normalization(data, mean, std) for data in dataset_train]

        # Train_loader
        train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
        torch.save(dict(loader = train_loader, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset_train.pt'))

        ### Preprocessing validation and test data

        # Apply normalization to the validation and test data
        dataset_val = [channel_wise_normalization(data, mean, std) for data in dataset_val]
        dataset_test = [channel_wise_normalization(data, mean, std) for data in dataset_test]

        val_loader = DataLoader(dataset_val, batch_size = 500, shuffle = False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(dataset_test, batch_size = 500, shuffle = False, collate_fn=custom_collate_fn)

        torch.save(dict(loader=val_loader, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset_val.pt'))
        torch.save(dict(loader=test_loader, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset_test.pt'))

    return train_loader, val_loader, test_loader, num_features, num_classes

### Define a function to load graph data and split the data for link prediction
### Our experiment settings follow https://github.com/tkipf/gae
def load_data_link(cfg):
    dataset = cfg.data.dataset
    path = cfg.data.data_root
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset = torch.load(osp.join(path, 'dataset_link.pt'))
        data = dataset['data']
        num_features = dataset['num_features']

    except FileNotFoundError:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dataset)
        elif dataset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dataset)
        elif dataset in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(path, dataset)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        num_features = dataset.num_features
        data = dataset[0]

        del data.train_mask
        del data.val_mask
        del data.test_mask

        split = T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0,
        )
        data.train_link_data, data.val_link_data, data.test_link_data = split(data)

        torch.save(dict(data=data, num_features=num_features),
                   osp.join(path, 'dataset_link.pt'))

    return data, num_features
