import os
import string
import numpy as np
import pandas as pd
import paddle
import paddle.nn.functional as F
from scipy.spatial.distance import squareform, pdist
from Bio import SeqIO
from biotite.structure.io import load_structure
from biotite.database import rcsb

from model.tokenizer import Alphabet
from finetune import ContactPredictionModel

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys.update({".": None, "*": None})
_translation = str.maketrans(deletekeys)

def read_msa(path):
    records = list(SeqIO.parse(path, "fasta"))
    return [(r.description, str(r.seq).translate(_translation)) for r in records]

def normalize(x, eps=1e-8):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)
    return x / norm

def extend(C, N, CA, L=1.522, A=1.927, D=-2.143):
    bc = normalize(CA - C)
    n = normalize(np.cross(N - C, bc))
    m = np.stack([bc, np.cross(n, bc), n], axis=-1)
    d = np.array([L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)])
    return C + np.matmul(m, d)

def contacts_from_pdb(structure, chain="A", threshold=8.0):
    mask = ~structure.hetero
    mask &= (structure.chain_id == chain)
    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]
    CB = extend(C, N, CA)
    dist = squareform(pdist(CB))
    contacts = (dist < threshold).astype(np.int64)
    np.fill_diagonal(contacts, 0)
    return contacts

def compute_precisions(predictions, targets, minsep=6, maxsep=None):
    if isinstance(predictions, np.ndarray):
        predictions = paddle.to_tensor(predictions)
    if isinstance(targets, np.ndarray):
        targets = paddle.to_tensor(targets)

    if len(predictions.shape) == 2:
        predictions = predictions.unsqueeze(0)
    if len(targets.shape) == 2:
        targets = targets.unsqueeze(0)

    batch_size, seqlen, _ = predictions.shape
    seqlen_range = paddle.arange(seqlen)
    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)

    valid_mask = (sep.abs() >= minsep) & (targets >= 0)
    if maxsep is not None:
        valid_mask &= (sep.abs() < maxsep)

    predictions = paddle.where(valid_mask, predictions, paddle.full_like(predictions, float('-inf')))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen
    indices = paddle.argsort(predictions_upper, axis=-1, descending=True)[:, :topk]
    batch_indices = paddle.arange(batch_size).unsqueeze(1).tile([1, topk])
    topk_targets = paddle.take_along_axis(targets_upper, indices, axis=1)

    if topk_targets.shape[1] < topk:
        pad_len = topk - topk_targets.shape[1]
        topk_targets = F.pad(topk_targets, [0, pad_len])

    cumulative_dist = paddle.cumsum(topk_targets, axis=-1)
    gather_indices = (paddle.arange(0.1, 1.1, 0.1) * topk).astype('int64') - 1
    gather_indices = paddle.clip(gather_indices, 0, topk - 1)
    binned_cumulative = paddle.take_along_axis(cumulative_dist, gather_indices.unsqueeze(0).expand([batch_size, -1]), axis=1)
    binned_precisions = binned_cumulative / (gather_indices + 1)

    return {
        "AUC": binned_precisions.mean(axis=-1),
        "P@L": binned_precisions[:, 9],
        "P@L2": binned_precisions[:, 4],
        "P@L5": binned_precisions[:, 1],
    }

def evaluate_prediction(pred_logits, targets):
    contact_ranges = [("local", 3, 6), ("short", 6, 12), ("medium", 12, 24), ("long", 24, None)]
    metrics = {}
    for name, minsep, maxsep in contact_ranges:
        stats = compute_precisions(pred_logits, targets, minsep, maxsep)
        for k, v in stats.items():
            metrics[f"{name}_{k}"] = v.item()
    return metrics

def evaluate_model_on_dataset(model_path, pdb_ids, msa_dir, device="gpu"):
    paddle.set_device(device)

    model = ContactPredictionModel()
    raw_state = paddle.load(model_path)
    model.set_state_dict(raw_state)
    model.eval()

    alphabet = Alphabet.from_architecture("esm-aa")
    batch_converter = alphabet.get_batch_converter()

    results = []
    for pdb_id in pdb_ids:
        print(f"Evaluating {pdb_id}...")

        structure = load_structure(f'data/eval/{pdb_id}.cif', extra_fields=["atom_id"])
        contact_map = contacts_from_pdb(structure, chain="A")

        msa = read_msa(os.path.join(msa_dir, f"{pdb_id.lower()}_1_A.a3m"))
        seq_id, seq = msa[0]
        _, _, tokens = batch_converter([(seq_id, seq)])
        tokens = tokens

        pred_logits = model(tokens)

        metrics = evaluate_prediction(pred_logits.cpu(), paddle.to_tensor(contact_map).unsqueeze(0))
        metrics.update({"pdb_id": pdb_id})
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index("pdb_id").T
    print("\nEvaluation Results:")
    print(df.to_string(float_format="%.4f"))

    return df

if __name__ == '__main__':
    model_ckpt_path = "checkpoints/finetune/finetune_step25000.pdparams"
    pdb_id_list = ["1a3a", "5ahw", "1xcr"]
    msa_folder = "data/eval"

    evaluate_model_on_dataset(model_ckpt_path, pdb_id_list, msa_folder)
