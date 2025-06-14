import os
import gzip
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist

from paddle.io import Dataset, DataLoader
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

from model.model import ESM2_AA
from model.tokenizer import Alphabet


def compute_distance_matrix(coords):
    coords = coords.astype('float32')
    mask = ~paddle.isnan(coords).any(axis=-1)
    coords = paddle.where(mask.unsqueeze(-1), coords, paddle.zeros_like(coords))
    dist = paddle.linalg.norm(coords.unsqueeze(2) - coords.unsqueeze(1), axis=-1)
    mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
    dist = paddle.where(mask_2d, dist, paddle.zeros_like(dist))
    return dist


def corrupt_coordinates(coords, epsilon=1.0):
    corrupted = coords.copy()
    for i in range(len(corrupted)):
        if not np.any(np.isnan(corrupted[i])):
            corrupted[i] += np.random.normal(0, epsilon, size=(3,))
    return corrupted


def safe_aa3_to_aa1(aa3):
    aa3 = aa3.capitalize()
    return protein_letters_3to1.get(aa3, 'X')


class ProteinMoleculeDataset(Dataset):
    def __init__(self, pdb_dir, config):
        self.config = config
        self.pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb.gz")]
        print(f"{len(self.pdb_files)} PDB files found in {pdb_dir}")
        self.alphabet = Alphabet.from_architecture("esm-aa")
        self.batch_converter = self.alphabet.get_batch_converter()

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        file = self.pdb_files[idx]
        parser = PDBParser(QUIET=True)
        with gzip.open(file, 'rt') as handle:
            structure = parser.get_structure("AF_model", handle)

        coords = []
        atom_names = []
        elements = []
        residue_seq = []
        residue_atom_map = {}
        residue_idx = 0

        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()
                    residue_seq.append(safe_aa3_to_aa1(resname))
                    start_idx = len(coords)
                    for atom in residue:
                        if atom.element == 'H':
                            continue
                        coords.append(atom.coord)
                        atom_names.append(atom.get_name())
                        elements.append(atom.element)
                    end_idx = len(coords)
                    residue_atom_map[residue_idx] = (start_idx, end_idx)
                    residue_idx += 1
            break

        unzip_ratio = self.config.get('residue_unzip_ratio', 0.01)
        L = len(residue_seq)
        unzip_indices = sorted(random.sample(range(L), max(1, int(L * unzip_ratio))))

        input_tokens, mlm_labels, coords_full = [], [], []
        atom_tokens = [f"{el.capitalize()}_a" for el in elements]

        for i in range(L):
            input_tokens.append(residue_seq[i])
            mlm_labels.append(residue_seq[i])
            coords_full.append([np.nan, np.nan, np.nan])
            if i in unzip_indices:
                a_start, a_end = residue_atom_map[i]
                input_tokens.extend(atom_tokens[a_start:a_end])
                mlm_labels.extend(atom_tokens[a_start:a_end])
                coords_full.extend(coords[a_start:a_end])

        _, _, input_ids = self.batch_converter([(os.path.basename(file), " ".join(input_tokens))])
        mlm_label_ids = [self.alphabet.get_idx(tok) for tok in mlm_labels]

        max_len = self.config.get("max_len", 1024)
        if len(input_tokens) > max_len:
            input_tokens = input_tokens[:max_len]
            mlm_labels = mlm_labels[:max_len]
            coords_full = coords_full[:max_len]

        _, _, input_ids = self.batch_converter([(os.path.basename(file), " ".join(input_tokens))])
        mlm_label_ids = [self.alphabet.get_idx(tok) for tok in mlm_labels]

        return {
            'input_ids': input_ids[0],
            'mlm_labels': paddle.to_tensor(mlm_label_ids, dtype='int64'),
            'coords': np.array(coords_full, dtype=np.float32),
            'coords_corrupt': corrupt_coordinates(np.array(coords_full, dtype=np.float32))
        }


class ESM2_AA_Model(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.model = ESM2_AA(
            num_layers=12,
            embed_dim=480,
            attention_heads=20,
            alphabet="ESM-AA",
            token_dropout=True,
            build_dist_head=True
        )

    def forward(self, batch):
        tokens = batch["input_ids"]
        dist_noisy = compute_distance_matrix(batch["coords_corrupt"])
        edge_type = paddle.zeros_like(dist_noisy, dtype='int64')
        out = self.model(tokens=tokens, src_distance=dist_noisy, src_edge_type=edge_type)
        dist_recon = self.model.dist_head(out["pair_rep"])
        return {"mlm_logits": out["logits"], "coords_recon": dist_recon}


class PolynomialLRWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, end_lr, power=1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power
        self.current_step = 0
        self.base_lrs = [group.optimize_attr['learning_rate'] for group in optimizer._param_groups]

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            scale = ((1 - (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)) ** self.power)
        for i, param_group in enumerate(self.optimizer._param_groups):
            param_group.optimize_attr['learning_rate'] = self.end_lr + (self.base_lrs[i] - self.end_lr) * scale


class PretrainESMAA:
    def __init__(self, config):
        self.config = config
        self.model = ESM2_AA_Model(config)
        self.model = paddle.DataParallel(self.model)

        self.optimizer = paddle.optimizer.AdamW(
            parameters=self.model.parameters(),
            learning_rate=config['lr'],
            beta1=0.9,
            beta2=0.98,
            weight_decay=config['weight_decay']
        )

        self.scheduler = PolynomialLRWarmup(
            self.optimizer,
            warmup_steps=config['warmup_steps'],
            total_steps=config['max_steps'],
            end_lr=config['end_lr']
        )

        self.criterion_mlm = nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_pdr = nn.SmoothL1Loss()

        self.train_loader = DataLoader(
            ProteinMoleculeDataset(config['train_data'], config),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn
        )

        self.ckpt_dir = config.get("ckpt_dir", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def collate_fn(self, batch):
        max_len = max(len(x['input_ids']) for x in batch)
        pad = lambda x, val: paddle.concat([x, paddle.full([max_len - len(x)], val, dtype=x.dtype)])
        pad_np = lambda x, val: np.pad(x, ((0, max_len - len(x)), (0, 0)), constant_values=val)
        return {
            "input_ids": paddle.stack([pad(item['input_ids'], 0) for item in batch]),
            "mlm_labels": paddle.stack([pad(item['mlm_labels'], 0) for item in batch]),
            "coords": paddle.to_tensor(np.stack([pad_np(item['coords'], np.nan) for item in batch]), dtype='float32'),
            "coords_corrupt": paddle.to_tensor(np.stack([pad_np(item['coords_corrupt'], np.nan) for item in batch]), dtype='float32')
        }

    def compute_losses(self, outputs, labels):
        logits = outputs['mlm_logits']
        loss_mlm = self.criterion_mlm(logits.reshape([-1, logits.shape[-1]]), labels['mlm_labels'].reshape([-1]))
        dist_true = compute_distance_matrix(labels['coords'])
        loss_pdr = self.criterion_pdr(outputs['coords_recon'], dist_true)
        loss_total = 4.0 * loss_mlm + 10.0 * loss_pdr
        return loss_total, loss_mlm, loss_pdr

    def save_checkpoint(self, step):
        ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_step{int(step)}.pdparams")
        paddle.save(self.model.state_dict(), ckpt_path)

    def train(self):
        self.model.train()
        step = 0
        total_steps = self.config['max_steps']

        pbar = tqdm(total=total_steps, desc="Pretraining", ncols=120)

        while step < total_steps:
            for batch in self.train_loader:
                outputs = self.model(batch)
                loss, loss_mlm, loss_pdr = self.compute_losses(outputs, batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                self.scheduler.step()

                pbar.set_postfix({
                    "Total": f"{loss.item():.4f}",
                    "MLM": f"{loss_mlm.item():.4f}",
                    "PDR": f"{loss_pdr.item():.4f}",
                    "Step": int(step)
                })
                pbar.update(int(self.config['batch_size'] / 4))

                if int(step) % self.config.get("save_every", 1000) == 0:
                    self.save_checkpoint(step)

                step += self.config['batch_size'] / 4
                if step >= total_steps:
                    break

        pbar.close()


if __name__ == '__main__':
    dist.init_parallel_env()

    config = {
        'train_data': 'data/swissprot',
        'batch_size': 4,
        'lr': 1e-4,
        'end_lr': 1e-5,
        'weight_decay': 1e-2,
        'warmup_steps': 5000,
        'max_steps': 300000,
        'save_every': 1000,
        'residue_unzip_ratio': 0.01,
        'distance_noise': 1.0,
        'ckpt_dir': 'checkpoints/train',
        'max_len': 1536,
    }
    trainer = PretrainESMAA(config)
    trainer.train()
