import streamlit as st
import torch
import dgl
import pickle
import re
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models import GNNBlocks, WGCN, MultiscaleCNN, GNNBlockDTI
from src.data.preprocessing import smi_2_graph, get_uv
from configs.biosnap import BIOSNAPConfig

st.set_page_config(page_title="DTI Prediction | GNNBlockDTI", layout="wide", page_icon="🧬")

# ── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height:100vh; }
.stButton>button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px; padding: 1.2rem; text-align:center;
}
.result-high { color: #43e97b; font-size: 1.3rem; font-weight: 700; }
.result-low  { color: #f5576c; font-size: 1.3rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Drug–Target Interaction Prediction")
st.markdown("#### Powered by **GNNBlockDTI** with Gated Protein Feature Fusion")

# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained model weights…")
def load_model():
    config = BIOSNAPConfig()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net_D  = GNNBlocks(input_size=64, hidden_size=96,
                       n_layer=config.n_layer, n_gnn=config.n_gnn,
                       dropout=config.dropout_d)
    net_T  = MultiscaleCNN(inputs=30, embed_dim=config.embed_dim,
                           nums_conv=[32, 64, 96], ksize=[5, 7, 13], dropout=0.2)
    net_T1 = WGCN(input_size=30, hidden_size=config.hidden_size, dropout=0.2)

    model = GNNBlockDTI(net_D, net_T, net_T1,
                        Drug_len=config.drug_len,
                        Target_len=config.target_len,
                        Target_len1=config.target_len1,
                        hid_size=config.hid_size,
                        dropout=config.dropout_DT,
                        fusion_type='gated')

    model_path = 'models/GNNBlockDTI_BIOSNAP_CV0.pt'
    weights_loaded = False
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        weights_loaded = True

    model.eval()
    return model, device, weights_loaded


# ── Load cached known-dataset features (optional, for fast lookup) ───────────
@st.cache_resource(show_spinner="Loading dataset features…")
def load_dataset_cache():
    try:
        with open('data/dataset/BIOSNAP/drug_graph.pkl', 'rb') as f:
            drug_graphs = pickle.load(f)
        with open('data/dataset/BIOSNAP/target_embedding.pkl', 'rb') as f:
            target_embeddings = pickle.load(f)
        with open('data/dataset/BIOSNAP/target_graph.pkl', 'rb') as f:
            target_graphs = pickle.load(f)
        return drug_graphs, target_embeddings, target_graphs
    except FileNotFoundError:
        return {}, {}, {}


# ── ProtBERT embedding for a single sequence (lazy-load) ─────────────────────
@st.cache_resource(show_spinner="Loading ProtBERT (one-time)…")
def load_protbert():
    from transformers import BertForMaskedLM, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    bert_model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd')
    bert_model.eval()
    return tokenizer, bert_model


def embed_sequence(seq: str):
    """Convert a raw amino-acid sequence to ProtBERT logit embedding [L, 30]."""
    tokenizer, bert_model = load_protbert()
    seq_spaced = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
    ids = tokenizer(seq_spaced, return_tensors='pt')
    with torch.no_grad():
        output = bert_model(**ids)
    embedding = output.logits[0][1:-1].cpu()          # [L, 30]
    assert embedding.shape[0] == len(seq), "Embedding length mismatch"
    return embedding


def seq_to_target_graph(embedding: torch.Tensor):
    """Build a protein contact graph with backbone-only edges (positive weights=1.0).
    Used for unseen proteins where ESM-1b maps are not precomputed."""
    n = embedding.shape[0]
    # Backbone edges: i→i+1 and i+1→i
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    weights = [1.0] * len(src)

    g = dgl.graph((src, dst), num_nodes=n)
    g.ndata['feat'] = embedding
    g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
    return g


# ── UI ───────────────────────────────────────────────────────────────────────
model, device, weights_loaded = load_model()
drug_graphs, target_embeddings, target_graphs = load_dataset_cache()

# Show status outside cached function
if weights_loaded:
    st.toast("✅ Trained weights loaded!", icon="✅")
else:
    st.warning("⚠️ Trained .pt file not found — using random weights.")

st.divider()

mode = st.radio("Input Mode", ["🆔 Known Dataset IDs (fast)", "🔬 Raw SMILES + Protein Sequence (any unseen drug/target)"],
                horizontal=True)

col1, col2 = st.columns(2, gap="large")

if mode.startswith("🆔"):
    with col1:
        st.subheader("💊 Drug")
        drug_id = st.text_input("Drug ID", "DB08533",
                                help="e.g. DB08533  (from BIOSNAP dataset)")
    with col2:
        st.subheader("🧬 Target")
        target_id = st.text_input("Target Protein UniProt ID", "P49862",
                                  help="e.g. P49862  (from BIOSNAP dataset)")

else:
    with col1:
        st.subheader("💊 Drug SMILES")
        smiles_input = st.text_area("SMILES string",
                                    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
                                    height=120, help="Paste any valid SMILES string")
    with col2:
        st.subheader("🧬 Protein Sequence")
        seq_input = st.text_area("Amino-acid sequence (single-letter code)",
                                 "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
                                 height=120, help="Standard 1-letter amino acid sequence")

st.divider()

if st.button("🔍 Predict Interaction", use_container_width=True):
    try:
        with st.spinner("Running forward pass through GNNBlockDTI…"):

            if mode.startswith("🆔"):
                # ── Fast lookup from preprocessed dataset ──────────────────────
                if drug_id not in drug_graphs:
                    st.error(f"Drug ID '{drug_id}' not found in dataset. Switch to **Raw SMILES** mode.")
                    st.stop()
                if target_id not in target_embeddings:
                    st.error(f"Target ID '{target_id}' not found in dataset. Switch to **Raw Sequence** mode.")
                    st.stop()

                d_graph  = drug_graphs[drug_id]
                t_seq    = target_embeddings[target_id].unsqueeze(0)    # [1, L, 30]
                t_graph  = target_graphs[target_id]

            else:
                # ── On-the-fly preprocessing for unseen drug + target ──────────
                smiles_clean = smiles_input.strip()
                seq_clean    = seq_input.strip().upper()

                if not smiles_clean:
                    st.error("Please enter a SMILES string.")
                    st.stop()
                if not seq_clean:
                    st.error("Please enter a protein sequence.")
                    st.stop()

                # Drug graph from SMILES
                with st.status("🔩 Converting SMILES → molecular graph…", expanded=False):
                    d_graph = smi_2_graph(smiles_clean)

                # Protein: ProtBERT embedding
                with st.status("🤖 Computing ProtBERT embedding for protein…", expanded=False):
                    t_embedding = embed_sequence(seq_clean)

                t_seq   = t_embedding.unsqueeze(0)          # [1, L, 30]
                t_graph = seq_to_target_graph(t_embedding)  # backbone-only graph

        # ── Forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            output = model((dgl.batch([d_graph]), t_seq, dgl.batch([t_graph])))
            probs  = torch.softmax(output, dim=-1)[0]
            prob_interact = probs[1].item()
            prob_no_inter = probs[0].item()

        st.divider()
        st.subheader("📊 Prediction Results")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;color:#aaa">Interaction Probability</div>
                <div style="font-size:2rem;font-weight:700;color:#667eea">{prob_interact:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;color:#aaa">No-Interaction Probability</div>
                <div style="font-size:2rem;font-weight:700;color:#764ba2">{prob_no_inter:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            verdict = "INTERACT" if prob_interact > 0.5 else "NO INTERACTION"
            color   = "#43e97b" if prob_interact > 0.5 else "#f5576c"
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:0.85rem;color:#aaa">Model Verdict</div>
                <div style="font-size:1.4rem;font-weight:700;color:{color}">{verdict}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        if prob_interact > 0.5:
            st.balloons()
            st.success(f"✅ **High probability of Drug–Target Interaction** (score = {prob_interact:.4f})")
        else:
            st.error(f"❌ **Low probability of Drug–Target Interaction** (score = {prob_interact:.4f})")

        if mode.startswith("🔬"):
            st.info("""ℹ️ **Note on unseen proteins:** The protein contact graph was built using 
            backbone-only edges (sequential residue connections) since ESM-1b contact maps 
            are not precomputed for unseen sequences. For highest accuracy, run the full 
            ESM-1b preprocessing pipeline to generate the contact map first.""")

    except ValueError as ve:
        st.error(f"Invalid input: {ve}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        raise e
