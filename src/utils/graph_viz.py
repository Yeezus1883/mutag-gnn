# import matplotlib.pyplot as plt
# import networkx as nx
# from torch_geometric.utils import to_networkx

# ATOM_MAP = {
#     0: "C",
#     1: "N",
#     2: "O",
#     3: "F",
#     4: "I",
#     5: "Cl",
#     6: "Br"
# }

# ATOM_COLOR = {
#     "C": "#909090",
#     "N": "#3050F8",
#     "O": "#FF0D0D",
#     "F": "#90E050",
#     "Cl": "#1FF01F",
#     "Br": "#A62929",
#     "I": "#940094"
# }


# def draw_molecule_graph(data):

#     G = to_networkx(data, to_undirected=True)

#     labels = {}
#     colors = []

#     for i, node_feat in enumerate(data.x):

#         atom_idx = node_feat.argmax().item()
#         atom = ATOM_MAP.get(atom_idx, "C")

#         labels[i] = atom
#         colors.append(ATOM_COLOR.get(atom, "#909090"))

#     pos = nx.spring_layout(G, seed=42)

#     fig, ax = plt.subplots(figsize=(5, 5))

#     nx.draw(
#         G,
#         pos,
#         ax=ax,
#         node_color=colors,
#         labels=labels,
#         node_size=800,
#         font_size=12,
#         font_weight="bold",
#         edge_color="black"
#     )

#     return fig


import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

ATOM_MAP = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "I",
    5: "Cl",
    6: "Br"
}

ATOM_COLOR = {
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094"
}


def draw_molecule_graph(data):

    G = to_networkx(data, to_undirected=True)

    labels = {}
    colors = []

    for i, node_feat in enumerate(data.x):

        atom_idx = node_feat.argmax().item()
        atom = ATOM_MAP.get(atom_idx, "C")

        labels[i] = atom
        colors.append(ATOM_COLOR.get(atom, "#909090"))

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(5,5))

    nx.draw(
        G,
        pos,
        ax=ax,
        node_color=colors,
        labels=labels,
        node_size=800,
        font_size=12,
        font_weight="bold"
    )

    return fig