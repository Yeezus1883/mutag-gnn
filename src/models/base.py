# from .gcn import GCN
# from .gin import GIN
# from .gat import GAT

# def get_model(config, in_channels, num_classes):

#     if config["model"] == "gcn":
#         return GCN(
#             in_channels=in_channels,
#             hidden_dim=config["hidden_dim"],
#             out_channels=num_classes,
#             dropout=config["dropout"]
#         )

#     elif config["model"] == "gin":
#         return GIN(
#             in_channels=in_channels,
#             hidden_dim=config["hidden_dim"],
#             out_channels=num_classes,
#             dropout=config["dropout"]
#         )
#     elif config["model"] == "gat":
#         return GAT(
#             in_channels=in_channels,
#             hidden_dim=config["hidden_dim"],
#             out_channels=num_classes,
#             heads=config["heads"],
#             dropout=config["dropout"]
#         )
#     else:
#         raise ValueError("Unsupported model")

from src.models.gcn import GCN
from src.models.gin import GIN
from src.models.gat import GAT


def get_model(config, in_channels, num_classes):

    model_name = config["model"]

    if model_name == "gcn":
        return GCN(in_channels, config["hidden_dim"], num_classes, config["dropout"])

    if model_name == "gin":
        return GIN(in_channels, config["hidden_dim"], num_classes, config["dropout"])

    if model_name == "gat":
        return GAT(in_channels, config["hidden_dim"], num_classes, config["dropout"])

    raise ValueError("Unknown model")