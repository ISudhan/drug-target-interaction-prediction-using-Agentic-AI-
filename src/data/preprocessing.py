"""
Data preprocessing pipeline for GNNBlockDTI.

Architecture specifies three main data components:
1. Drug Molecular Graph (from SMILES)
2. Protein Sequence Representation (from ProtBERT)
3. Protein Contact Map Graph (from ESM-1b)
"""

import math
import re
import dgl
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from transformers import BertForMaskedLM, BertTokenizer
import esm


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


def target_graph_construct(proteins, device=None):
    """Construct contact map probabilities using ESM-1b.
    
    Handles sequences > 1000 length by processing in chunks of 500 with overlap.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()
    
    target_distance = {}
    
    for pro_id, seq in tqdm(proteins.items(), desc="ESM-1b Contact Maps"):
        data = [(pro_id, seq)]
        
        # Original code processed short sequences on GPU but long sequences on CPU.
        # We process everything on the specified device.
        if len(seq) <= 1000:
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0]
            target_distance[pro_id] = contact_map.cpu().numpy()
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))
            interval = 500
            chunks = math.ceil(len(seq) / interval)
            
            for s in range(chunks):
                start = s * interval
                end = min((s + 2) * interval, len(seq))
                
                temp_seq = seq[start:end]
                temp_data = [(pro_id, temp_seq)]
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                
                with torch.no_grad():
                    results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
                    
                # Update global contact map with overlap averaging
                chunk_contacts = results["contacts"][0].cpu().numpy()
                
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                # Add offset
                row_global = row + start
                col_global = col + start
                
                contact_prob_map[start:end, start:end] += chunk_contacts
                contact_prob_map[row_global, col_global] /= 2.0
                
                if end == len(seq):
                    break
                    
            target_distance[pro_id] = contact_prob_map

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
