import os
import sys
import argparse
import pickle
import torch
import gc   

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import (
    get_drug_graph, 
    get_protbert_embedding,
    target_graph_construct, 
    get_target_graph
)

def main():
    parser = argparse.ArgumentParser(description="Preprocess BIOSNAP raw data")
    parser.add_argument("--task", type=str, default="BIOSNAP", help="Dataset/Task name")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for embeddings")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if files exist")
    args = parser.parse_args()
    
    data_dir = args.data_dir if args.data_dir else os.path.join("data/dataset", args.task)
    device = torch.device(args.device)
    
    print(f"=== BIOSNAP Preprocessing ===")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    
    # Load raw data
    smi_path = os.path.join(data_dir, "drug_smi_raw.pkl")
    seq_path = os.path.join(data_dir, "prot_seq_raw.pkl")
    
    try:
        with open(smi_path, 'rb') as f:
            drugs = pickle.load(f)
        with open(seq_path, 'rb') as f:
            proteins = pickle.load(f)
        print(f"Loaded {len(drugs)} drugs and {len(proteins)} proteins.")
    except FileNotFoundError as e:
        print(f"Raw data files not found: {e}")
        sys.exit(1)
        
    # 1. Drug Graphs
    drug_out = os.path.join(data_dir, "drug_graph.pkl")
    if not os.path.exists(drug_out) or args.force:
        print("\n--- Step 1: Drug Graphs ---")
        drug_graph = get_drug_graph(drugs)
        with open(drug_out, 'wb') as f:
            pickle.dump(drug_graph, f)
        print(f"Saved {len(drug_graph)} drug graphs to {drug_out}")
    else:
        print(f"\n--- Step 1: Drug Graphs --- [SKIPPED, exists]")
        with open(drug_out, 'rb') as f:
            drug_graph = pickle.load(f)

    # 2. ProtBERT Embeddings
    protbert_out = os.path.join(data_dir, "target_embedding.pkl")
    if not os.path.exists(protbert_out) or args.force:
        print("\n--- Step 2: ProtBERT Embeddings ---")
        target_embedding = get_protbert_embedding(proteins, device=device)
        with open(protbert_out, 'wb') as f:
            pickle.dump(target_embedding, f)
        print(f"Saved {len(target_embedding)} embeddings to {protbert_out}")
    else:
        print(f"\n--- Step 2: ProtBERT Embeddings --- [SKIPPED, exists]")
        with open(protbert_out, 'rb') as f:
            target_embedding = pickle.load(f)
            
    if str(device) != 'cpu':
        torch.cuda.empty_cache()
        gc.collect()

    # 3. ESM-1b Contact Maps
    dist_out = os.path.join(data_dir, "target_distance.pkl")

    print("\n--- Step 3: ESM-1b Contact Maps ---")

    # Always call target_graph_construct().
    # It automatically resumes from target_distance.pkl.
    target_distance = target_graph_construct(
        proteins,
        device=device,
        output_path=dist_out,
        save_every=10
    )

    with open(dist_out, 'wb') as f:
        pickle.dump(target_distance, f)

    print(f"Saved {len(target_distance)} contact maps to {dist_out}")
            
    if str(device) != 'cpu':
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Target DGL Graphs
    target_graph_out = os.path.join(data_dir, "target_graph.pkl")
    if not os.path.exists(target_graph_out) or args.force:
        print("\n--- Step 4: Target DGL Graphs ---")
        target_graph = get_target_graph(target_embedding, target_distance)
        with open(target_graph_out, 'wb') as f:
            pickle.dump(target_graph, f)
        print(f"Saved {len(target_graph)} target graphs to {target_graph_out}")
    else:
        print(f"\n--- Step 4: Target DGL Graphs --- [SKIPPED, exists]")
        
    print("\n✅ Preprocessing Complete!")

if __name__ == "__main__":
    main()
