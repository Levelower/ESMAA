import os
import re
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

from paddle.io import Dataset, DataLoader
from tqdm import tqdm

from model.model import ESM2_AA
from model.tokenizer import Alphabet


class ContactPredictionDataset(Dataset):
    def __init__(self, sequences, contact_maps, alphabet):
        self.sequences = sequences
        self.contact_maps = contact_maps
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        contacts = self.contact_maps[idx]
        _, _, tokens = self.batch_converter([(str(idx), seq)])
        return {
            'tokens': tokens[0],
            'contact_map': paddle.to_tensor(contacts, dtype='float32')
        }


class ContactPredictionModel(nn.Layer):
    def __init__(self, pretrained_ckpt=None):
        super().__init__()
        self.encoder = ESM2_AA(
            num_layers=12,
            embed_dim=480,
            attention_heads=20,
            alphabet="ESM-AA",
            token_dropout=False,
            build_dist_head=False
        )

        if pretrained_ckpt:
            print(f"Loading pretrained weights from: {pretrained_ckpt}")
            raw_state = paddle.load(pretrained_ckpt)
            state_dict = {}
            for k, v in raw_state.items():
                new_key = k.replace('model.', '')
                state_dict[new_key] = v
            self.encoder.set_state_dict(state_dict)

    def forward(self, tokens):
        out = self.encoder(tokens=tokens, return_contacts=True)
        return out["contacts"]


class ContactFineTuner:
    def __init__(self, model, dataset, config):
        self.config = config
        self.model = paddle.DataParallel(model)

        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=self.collate_fn,
            return_list=True
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(), learning_rate=config['lr']
        )
        self.ckpt_dir = config.get("ckpt_dir", "checkpoints/finetune")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def collate_fn(self, batch):
        max_len = self.config.get('max_len', None)  # 截断长度（不包括 BOS/EOS）
        result_tokens = []
        result_contacts = []

        for item in batch:
            tokens = item['tokens']  # 含 BOS/EOS，shape: (L + 2,)
            contact = item['contact_map']  # shape: (L, L)

            L = contact.shape[0]  # 不含 BOS/EOS 的有效长度

            if max_len is not None and L > max_len:
                # tokens = [BOS] + residues + [EOS]，保留 max_len 个 residue
                tokens = paddle.concat([tokens[:1], tokens[1:1 + max_len], tokens[-1:]])  # shape: (max_len + 2,)
                contact = contact[:max_len, :max_len]  # shape: (max_len, max_len)

            result_tokens.append(tokens)
            result_contacts.append(contact)

        max_len_tokens = max([x.shape[0] for x in result_tokens])
        max_len_contacts = max([x.shape[0] for x in result_contacts])

        def pad_tokens(x, val=0):
            pad_len = max_len_tokens - x.shape[0]
            pad = paddle.full([pad_len], val, dtype=x.dtype)
            return paddle.concat([x, pad])

        def pad_matrix(m, val=0):
            pad_len = max_len_contacts - m.shape[0]
            return F.pad(m, [0, pad_len, 0, pad_len], mode='constant', value=val)

        batch_tokens = paddle.stack([pad_tokens(x) for x in result_tokens])
        batch_contacts = paddle.stack([pad_matrix(x) for x in result_contacts])

        return {
            'tokens': batch_tokens,              # shape: [B, T+2]
            'contact_maps': batch_contacts       # shape: [B, T, T]
        }



    def save_checkpoint(self, step):
        paddle.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f"finetune_step{int(step)}.pdparams"))

    def train(self):
        self.model.train()
        total_steps = self.config['epochs'] * len(self.dataloader)
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        step = 0
        for epoch in range(self.config['epochs']):
            for batch in self.dataloader:
                tokens = batch['tokens']
                labels = batch['contact_maps']
                logits = self.model(tokens)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                if step % self.config['log_every'] == 0:
                    tqdm.write(f"[Step {step}] Loss: {loss.item():.4f}")

                if step % self.config['save_every'] == 0:
                    self.save_checkpoint(step)

                pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=epoch + 1)
                pbar.update(1)
                step += 1
        pbar.close()


def extend(C, N, CA, L=1.522, A=1.927, D=-2.143):
    def normalize(x, eps=1e-8):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        norm = np.where(norm < eps, 1.0, norm)
        return x / norm

    bc = normalize(CA - C)
    n = normalize(np.cross(N - C, bc))
    m = np.stack([bc, np.cross(n, bc), n], axis=-1)
    d = np.array([L * np.cos(A),
                  L * np.sin(A) * np.cos(D),
                 -L * np.sin(A) * np.sin(D)])
    Cbeta = C + np.matmul(m, d)
    return Cbeta


def parse_casp_file(filepath, distance_threshold=8.0):
    sequences, contact_maps = [], []

    with open(filepath, 'r') as f:
        content = f.read()

    entries = content.strip().split('[ID]')

    for entry in entries:
        if not entry.strip():
            continue

        seq_match = re.search(r"\[PRIMARY\]\n([A-Z\n]+)", entry)
        tertiary_match = re.search(r"\[TERTIARY\]\n([\d\.\-\s\n]+)", entry)

        if not seq_match or not tertiary_match:
            continue

        seq = seq_match.group(1).replace("\n", "")
        L = len(seq)

        try:
            tertiary_all = list(map(float, tertiary_match.group(1).split()))
        except:
            continue

        if len(tertiary_all) != 3 * 3 * L:
            continue

        coords = np.array(tertiary_all).reshape(3, 3 * L).T

        try:
            N_coords  = coords[0::3]
            CA_coords = coords[1::3]
            C_coords  = coords[2::3]
        except:
            continue

        Cbeta_coords = extend(C=C_coords, N=N_coords, CA=CA_coords)

        dist = np.linalg.norm(Cbeta_coords[:, None, :] - Cbeta_coords[None, :, :], axis=-1)
        contact_map = (dist < distance_threshold).astype(np.float32)
        np.fill_diagonal(contact_map, 0)

        sequences.append(seq)
        contact_maps.append(contact_map)

    return sequences, contact_maps


if __name__ == '__main__':
    dist.init_parallel_env()

    sequences, contact_maps = parse_casp_file("data/casp7/training_30")

    alphabet = Alphabet.from_architecture("esm-aa")

    dataset = ContactPredictionDataset(sequences, contact_maps, alphabet)
    model = ContactPredictionModel(pretrained_ckpt="checkpoints/train/ckpt_step200000.pdparams")

    config = {
        'epochs': 5,
        'batch_size': 2,
        'lr': 1e-4,
        'log_every': 100,
        'save_every': 1000,
        'log_dir': "runs/finetune",
        'ckpt_dir': "checkpoints/finetune",
        'max_len': 1048
    }

    trainer = ContactFineTuner(model, dataset, config)
    trainer.train()

