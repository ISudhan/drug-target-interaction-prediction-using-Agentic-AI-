"""
Data preprocessing pipeline for GNNBlockDTI.

Architecture specifies three main data components:
1. Drug Molecular Graph (from SMILES)
2. Protein Sequence Representation (from ProtBERT)
3. Protein Contact Map Graph (from ESM-1b)
"""

import math
import re
import gc
import dgl
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import os
import pickle

# NOTE: transformers (BertForMaskedLM, BertTokenizer) and esm are imported
# lazily inside get_protbert_embedding() and target_graph_construct() to avoid
# requiring these heavy packages for drug graph construction.


def one_hot(char, dictionary):
    """One-hot encoding helper."""
    h = torch.zeros(len(dictionary) + 1)
    for i in range(len(dictionary)):
        if dictionary[i] == char:
            h[i] = 1
            break
        elif i == len(dictionary) - 1:
            h[-1] = 1
    return h


def smi_2_graph(smi):
    """Converts a SMILES string to a bidirected DGL graph with 64-dim atom features.

    Features (64 total):
    - Atom symbol (43 + 1)
    - Formal charge (8 + 1)
    - Degree (8 + 1)
    - Is Aromatic (1)
    - Is In Ring (1)
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    nums_atom = mol.GetNumAtoms()

    u, v = [], []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        u.append(begin.GetIdx())
        v.append(end.GetIdx())

    u, v = torch.tensor(u), torch.tensor(v)
    g = dgl.graph((u, v))

    feats = []
    # Official paper atom symbols
    symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
               'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
               'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
               'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']

    for atom in mol.GetAtoms():
        feat_0 = one_hot(atom.GetSymbol(), symbols) # 44
        feat_1 = one_hot(atom.GetFormalCharge(), [1, 2, 3, 4, 5, 6, 7, 8]) # 9
        feat_2 = one_hot(atom.GetDegree(), [1, 2, 3, 4, 5, 6, 7, 8]) # 9
        feat_3 = torch.tensor([float(atom.GetIsAromatic())]) # 1
        feat_4 = torch.tensor([float(atom.IsInRing())]) # 1

        # 44 + 9 + 9 + 1 + 1 = 64 dimensional feature
        feat = torch.cat([feat_0, feat_1, feat_2, feat_3, feat_4])
        feats.append(feat)

    feats_1 = torch.cat([feats[idx].unsqueeze(0) for idx in range(len(feats))], 0)

    # Handle single-atom molecules edge case
    if len(u) > 0 and len(v) > 0:
        nums_n = min(max(max(u), max(v)) + 1, nums_atom)
    else:
        nums_n = nums_atom

    feats_2 = feats_1[:nums_n]

    # Graph is bidirected
    bg = dgl.to_bidirected(g)
    bg.ndata['feat'] = feats_2

    return bg


def get_drug_graph(ligands):
    """Batch convert SMILES to drug graphs."""
    XD = {}
    for d in tqdm(ligands.keys(), desc="Drug Graphs"):
        try:
            XD[str(d)] = smi_2_graph(ligands[d])
        except torch.cuda.OutOfMemoryError:
            print(
                f"\nCUDA OOM while processing {pro_id} "
                f"(length={len(seq)})"
            )

            # Save every successfully processed protein
            # before the failed protein.
            if output_path is not None:
                with open(output_path, "wb") as f:
                    pickle.dump(target_distance, f)

                print(
                    f"Saved checkpoint before OOM: "
                    f"{len(target_distance)} proteins"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

            raise

        except Exception as e:
            print(f"Failed to process SMILES for {d}: {ligands[d]} - {e}")

    return XD


def get_protbert_embedding(data, device=None):
    """Extract protein embeddings from ProtBERT.

    CRITICAL: The official implementation uses the LOGITS output (dim=30),
    not the hidden states, and excludes the CLS/SEP tokens.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from transformers import BertForMaskedLM, BertTokenizer

    embed = dict()
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd").to(device)
    model.eval()

    for pid, seq in tqdm(data.items(), desc="ProtBERT Embeddings"):
        # Replace non-standard amino acids with 'X'
        seq1 = " ".join(list(re.sub(r"[UZOB]", "X", seq)))

        ids = tokenizer(seq1, return_tensors='pt')
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        # output[0] are the logits: [batch, seq_len+2, vocab_size]
        # vocab_size is 30. [1:-1] removes [CLS] and [SEP] tokens.
        output1 = output[0][0][1: -1]

        assert len(seq) == len(output1), f"Length mismatch: {len(seq)} != {len(output1)}"

        embed[pid] = output1.cpu()

        # Memory optimization
        del output, output1, input_ids, attention_mask, ids

    if str(device) != 'cpu':
        torch.cuda.empty_cache()
        gc.collect()

    return embed


def padding(data, maxlen):
    """Pad sequence embeddings to maxlen."""
    data_new = dict()
    for pid, vec in tqdm(data.items(), desc="Padding"):
        vec_new = torch.zeros((maxlen, vec.shape[1]), dtype=torch.float32)
        if vec.shape[0] < maxlen:
            vec_new[:vec.shape[0]] = vec[:]
        else:
            vec_new[:] = vec[:maxlen]
        data_new[pid] = vec_new

    return data_new


def target_graph_construct(proteins, device=None, output_path=None, save_every=10):
    """Construct ESM-1b contact maps incrementally.

    Long proteins are processed in overlapping chunks to reduce peak memory.
    Results are periodically saved so preprocessing can resume after interruption.

    Args:
        proteins: dict {protein_id: sequence}
        device: torch device
        output_path: optional path for incremental checkpoint
        save_every: save progress every N proteins
    """
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    import esm

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    model.to(device)
    model.eval()

    target_distance = {}

    # Resume previous progress if available
    if output_path is not None and os.path.exists(output_path):
        print(f"Resuming contact maps from: {output_path}")
        with open(output_path, "rb") as f:
            target_distance = pickle.load(f)

        print(f"Already processed: {len(target_distance)} proteins")

    remaining = [
        (pro_id, seq)
        for pro_id, seq in proteins.items()
        if pro_id not in target_distance
    ]

    print(f"Remaining proteins: {len(remaining)}")

    for idx, (pro_id, seq) in enumerate(
        tqdm(remaining, desc="ESM-1b Contact Maps"),
        start=1
    ):

        try:
            # ---------------------------------------------------------
            # Short proteins
            # ---------------------------------------------------------
            if len(seq) <= 500:

                data = [(pro_id, seq)]

                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(
                        batch_tokens,
                        repr_layers=[33],
                        return_contacts=True
                    )

                contact_map = results["contacts"][0].detach().cpu().numpy()

                target_distance[pro_id] = contact_map

                del batch_labels
                del batch_strs
                del batch_tokens
                del results
                del contact_map

            # ---------------------------------------------------------
            # Long proteins
            # ---------------------------------------------------------
            else:

                seq_len = len(seq)

                contact_prob_map = np.zeros(
                    (seq_len, seq_len),
                    dtype=np.float32
                )

                count_map = np.zeros(
                    (seq_len, seq_len),
                    dtype=np.float32
                )

                chunk_size = 500
                overlap = 250
                step = chunk_size - overlap

                starts = list(range(0, seq_len, step))

                for start in starts:

                    end = min(start + chunk_size, seq_len)

                    temp_seq = seq[start:end]

                    temp_data = [(pro_id, temp_seq)]

                    batch_labels, batch_strs, batch_tokens = \
                        batch_converter(temp_data)

                    batch_tokens = batch_tokens.to(device)

                    with torch.no_grad():
                        results = model(
                            batch_tokens,
                            repr_layers=[33],
                            return_contacts=True
                        )

                    chunk_contacts = (
                        results["contacts"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    chunk_len = end - start

                    contact_prob_map[
                        start:end,
                        start:end
                    ] += chunk_contacts[:chunk_len, :chunk_len]

                    count_map[
                        start:end,
                        start:end
                    ] += 1.0

                    # Free GPU memory immediately
                    del batch_labels
                    del batch_strs
                    del batch_tokens
                    del results
                    del chunk_contacts
                    del temp_data

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    gc.collect()

                    if end == seq_len:
                        break

                # Average overlapping regions
                mask = count_map > 0

                contact_prob_map[mask] /= count_map[mask]

                target_distance[pro_id] = contact_prob_map

                del contact_prob_map
                del count_map
                del mask

            # ---------------------------------------------------------
            # Periodic checkpoint
            # ---------------------------------------------------------
            if (
                output_path is not None
                and idx % save_every == 0
            ):
                print(
                    f"\nSaving progress: "
                    f"{len(target_distance)} proteins"
                )

                with open(output_path, "wb") as f:
                    pickle.dump(
                        target_distance,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            gc.collect()

        except RuntimeError as e:

            if "out of memory" in str(e).lower():

                print(
                    f"\nCUDA OOM while processing {pro_id} "
                    f"(length={len(seq)})"
                )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                gc.collect()

                # Save everything completed so far
                if output_path is not None:
                    print("Saving progress before aborting...")

                    with open(output_path, "wb") as f:
                        pickle.dump(
                            target_distance,
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL
                        )

                raise

            raise

    # Final save
    if output_path is not None:
        print(
            f"\nSaving final contact maps: "
            f"{len(target_distance)} proteins"
        )

        with open(output_path, "wb") as f:
            pickle.dump(
                target_distance,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return target_distance


def get_uv(adj):
    """Extract graph edges from contact map probability matrix.

    Uses threshold of > 0.5 for contact edges.
    Always includes sequential backbone edges (i-1, i+1) with weight 1.0.
    """
    u, v = [], []
    weight = []
    m, n = len(adj), len(adj[0])

    for i in range(m):
        for j in range(n):
            # Sequential edges always exist with weight 1.0
            if j == i - 1 or j == i + 1:
                u.append(i)
                v.append(j)
                weight.append(1.0)
                continue

            # Contact edges based on ESM probability threshold
            if adj[i][j] > 0.5:
                u.append(i)
                v.append(j)
                weight.append(adj[i][j])

    return u, v, weight


def get_target_graph(data, distance):
    """Construct target protein DGL graphs with ProtBERT features and ESM edges."""
    Gs = dict()
    for pid, feats in tqdm(data.items(), desc="Protein Graphs"):
        contact_map = distance[pid]
        u, v, weight = get_uv(contact_map)

        g = dgl.graph((u, v), num_nodes=len(feats))
        g.ndata["feat"] = feats
        g.edata["weight"] = torch.tensor(weight, dtype=torch.float32)
        Gs[pid] = g

    return Gs
