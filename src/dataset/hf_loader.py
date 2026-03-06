# import torch
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from datasets import load_dataset
# from sklearn.model_selection import train_test_split


# def load_mutag_from_hf(batch_size=32, seed=42):

#     hf_dataset = load_dataset("graphs-datasets/MUTAG")["train"]

#     data_list = []

#     for item in hf_dataset:

#         edge_index = torch.tensor(item["edge_index"], dtype=torch.long)
#         x = torch.tensor(item["node_feat"], dtype=torch.float)
#         edge_attr = torch.tensor(item["edge_attr"], dtype=torch.float)
#         y = torch.tensor(item["y"], dtype=torch.long)

#         graph = Data(
#             x=x,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             y=y
#         )

#         data_list.append(graph)

#     train_data, test_data = train_test_split(
#         data_list,
#         test_size=0.2,
#         random_state=seed,
#         stratify=[g.y.item() for g in data_list]
#     )

#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size)
#     in_channels = data_list[0].x.shape[1]
#     num_classes = 2

#     return train_loader, test_loader, in_channels, num_classes

import torch
from torch_geometric.data import Data
from datasets import load_dataset

def load_mutag_from_hf():

    hf_dataset = load_dataset("graphs-datasets/MUTAG")["train"]

    data_list = []

    for item in hf_dataset:

        edge_index = torch.tensor(item["edge_index"], dtype=torch.long)
        x = torch.tensor(item["node_feat"], dtype=torch.float)
        edge_attr = torch.tensor(item["edge_attr"], dtype=torch.float)
        y = torch.tensor(item["y"], dtype=torch.long)

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        data_list.append(graph)

    in_channels = data_list[0].x.shape[1]
    num_classes = 2

    return data_list, in_channels, num_classes