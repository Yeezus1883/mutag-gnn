import streamlit as st
import torch
import pandas as pd
import json

from src.dataset.hf_loader import load_mutag_from_hf
from src.models.base import get_model
from src.utils.graph_viz import draw_molecule_graph
from src.utils.smiles_to_graph import smiles_to_graph
import streamlit as st

with open(".streamlit/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

st.set_page_config(page_title="Mutagenicity Predictor", layout="wide")

st.title("Mutagenicity Predictor (Graph Neural Networks)")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    with open("experiments/best_result.json") as f:
        data = json.load(f)

    config = data["config"]

    data_list, in_channels, num_classes = load_mutag_from_hf()

    checkpoint = torch.load("experiments/best_model.pt", map_location=DEVICE)

    hidden_dim = checkpoint["conv1.nn.0.weight"].shape[0]

    config["hidden_dim"] = hidden_dim

    model = get_model(
        config=config,
        in_channels=in_channels,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model, config, data_list


model, config, data_list = load_model()
# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.markdown("## ⍟ Navigation ##")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Experiments",
        "Molecule Explorer"
    ],
    label_visibility="collapsed"

)    
st.divider()
st.caption("Dataset: MUTAG")
st.caption("Models: GCN / GIN / GAT")


# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    st.write("""
This project explores the use of ***Graph Neural Networks (GNNs)*** to predict whether a molecule is ***mutagenic or non-mutagenic*** using the MUTAG dataset, a benchmark dataset consisting of 188 chemical compounds.
             Three GNN architectures were implemented and compared:

Graph Convolutional Network (GCN)

Graph Isomorphism Network (GIN)

Graph Attention Network (GAT)

The models were trained using 10-fold cross-validation, allowing robust evaluation across multiple splits of the dataset. A Streamlit-based interactive dashboard was developed to visualize molecular graphs, run predictions, and analyze experimental results.
""")

    st.subheader("Best Model Configuration")

    config_df = pd.DataFrame(
        list(config.items()),
        columns=["Parameter", "Value"]
    )

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Experiments
# --------------------------------------------------

elif page == "Experiments":
    st.write("""The table below summarizes the results of all experiments conducted with different GNN architectures and hyperparameter settings. Each row corresponds to a unique experiment configuration, along with its mean accuracy and standard deviation across the 10 folds of cross-validation. Use this table to compare the performance of different models and identify trends in how hyperparameters affect accuracy.""")
    st.divider()
    st.header("Experiment Results")

    df = pd.read_csv("experiments/experiment_log.csv")

    st.dataframe(df)

    st.subheader("Model Accuracy Comparison")

    st.bar_chart(df.groupby("model")["accuracy_mean"].mean())

# --------------------------------------------------
# Molecule Explorer
# --------------------------------------------------

elif page == "Molecule Explorer":
    st.write("""This interactive tool allows you to explore the MUTAG dataset and make predictions on new molecules using the trained GNN model. In the first tab, you can select any molecule from the MUTAG dataset to visualize its structure and see the model's prediction for mutagenicity. In the second tab, you can upload your own CSV file containing a 'smiles' column with SMILES strings representing molecules. The app will predict whether each molecule is mutagenic or non-mutagenic, display the results in a table, and allow you to inspect individual molecules along with their predictions.""")
    st.header("Molecule Explorer")

    tab1, tab2 = st.tabs([
        "MUTAG Dataset Explorer",
        "SMILES Predictor"
    ])

    # ==================================================
    # MUTAG DATASET EXPLORER
    # ==================================================

    with tab1:

        st.subheader("Explore MUTAG Dataset")

        idx = st.slider(
            "Select Molecule",
            0,
            len(data_list)-1,
            0
        )

        graph = data_list[idx]

        col1, col2 = st.columns([2,1])

        with col1:

            fig = draw_molecule_graph(graph)

            st.pyplot(fig)

            st.write(f"Atoms: {graph.num_nodes}")
            st.write(f"Bonds: {graph.num_edges}")

        with col2:

            graph = graph.to(DEVICE)

            with torch.no_grad():

                batch = torch.zeros(
                    graph.x.shape[0],
                    dtype=torch.long,
                    device=DEVICE
                )

                out = model(graph.x, graph.edge_index, batch)

                prob = torch.softmax(out, dim=1)

                pred = prob.argmax().item()

            label = "Mutagenic ⚠️" if pred == 1 else "Non-Mutagenic ✅"

            st.metric("Prediction", label)

            st.metric(
                "Confidence",
                f"{prob.max().item():.3f}"
            )

    # ==================================================
    # SMILES DATASET PREDICTOR
    # ==================================================

    with tab2:

        st.subheader("Upload SMILES Dataset")

        uploaded_file = st.file_uploader(
            "Upload CSV containing a 'smiles' column",
            type=["csv"]
        )

        if uploaded_file:

            df = pd.read_csv(uploaded_file)

            st.write("Dataset Preview")

            st.dataframe(df.head())

            results = []
            graphs = []

            for _, row in df.iterrows():

                smiles = row["smiles"]

                graph = smiles_to_graph(smiles)

                if graph is None:
                    continue

                graph = graph.to(DEVICE)

                batch = torch.zeros(
                    graph.x.shape[0],
                    dtype=torch.long,
                    device=DEVICE
                )

                with torch.no_grad():

                    out = model(graph.x, graph.edge_index, batch)

                    prob = torch.softmax(out, dim=1)

                    pred = prob.argmax().item()

                label = "Mutagenic ⚠️" if pred == 1 else "Non-Mutagenic ✅"

                results.append({
                    "smiles": smiles,
                    "prediction": label,
                    "confidence": float(prob.max())
                })

                graphs.append(graph.cpu())

            results_df = pd.DataFrame(results)

            st.subheader("Prediction Results")

            st.dataframe(results_df)

            if len(graphs) > 0:

                idx = st.slider(
                    "Inspect Molecule",
                    0,
                    len(graphs)-1,
                    0
                )

                col1, col2 = st.columns([2,1])

                with col1:

                    fig = draw_molecule_graph(graphs[idx])

                    st.pyplot(fig)

                with col2:

                    st.metric(
                        "Prediction",
                        results_df.iloc[idx]["prediction"]
                    )

                    st.metric(
                        "Confidence",
                        f"{results_df.iloc[idx]['confidence']:.3f}"
                    )

            st.download_button(
                "Download Predictions",
                results_df.to_csv(index=False),
                file_name="mutagenicity_predictions.csv"
            )