import itertools
from typing import Sequence, Tuple, List, Union

import paddle


proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}

RawMSA = Sequence[Tuple[str, str]]


class Alphabet:
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks) + list(self.append_toks) + ["<null_1>"] + list(
            self.standard_toks
        )

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.tok_to_idx["<pad>"]
        self.cls_idx = self.tok_to_idx["<cls>"]
        self.mask_idx = self.tok_to_idx["<mask>"]
        self.eos_idx = self.tok_to_idx["<eos>"]

        self.all_special_tokens = ["<eos>", "<unk>", "<pad>", "<cls>", "<mask>"]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok: str):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx: int):
        return self.all_toks[idx]

    def to_dict(self):
        return self.tok_to_idx.copy()

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        name = name.lower()
        if name in ("esm-1", "protein_bert_base"):
            p_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            a_toks = ("<cls>", "<mask>", "<sep>")
            p_bos, a_eos, msa_flag = True, False, False
        elif name in ("esm-1b", "roberta_large"):
            p_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            a_toks = ("<mask>",)
            p_bos, a_eos, msa_flag = True, True, False
        elif name in ("msa transformer", "msa_transformer"):
            p_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            a_toks = ("<mask>",)
            p_bos, a_eos, msa_flag = True, False, True
        elif "invariant_gvp" in name:
            p_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            a_toks = ("<mask>", "<cath>", "<af2>")
            p_bos, a_eos, msa_flag = True, False, False
        elif "esm-aa" in name:
            p_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            a_toks = (
                "<mask>", "C_a", "N_a", "O_a", "S_a", "H_a", "Cl_a", "F_a", "Br_a", "I_a",
                "Si_a", "P_a", "B_a", "Na_a", "K_a", "Al_a", "Ca_a", "Sn_a",
                "As_a", "Hg_a", "Fe_a", "Zn_a", "Cr_a", "Se_a", "Gd_a", "Au_a", "Li_a",
            )
            p_bos, a_eos, msa_flag = True, True, False
        else:
            raise ValueError(f"Unknown architecture: {name}")

        return cls(
            proteinseq_toks["toks"], p_toks, a_toks, p_bos, a_eos, msa_flag
        )

    def _basic_tokenize(self, text: str):
        return text.split()

    def tokenize(self, text: str) -> List[str]:
        def split_on_token(tok, txt):
            out = []
            parts = txt.split(tok)
            for i, sub in enumerate(parts):
                if i < len(parts) - 1:
                    sub = sub.rstrip()
                if i > 0:
                    sub = sub.lstrip()
                if i == 0 and not sub:
                    out.append(tok)
                elif i == len(parts) - 1:
                    if sub:
                        out.append(sub)
                else:
                    if sub:
                        out.append(sub)
                    out.append(tok)
            return out

        def split_on_tokens(tok_list, txt):
            if not txt.strip():
                return []
            text_segments = [txt]
            for tok in tok_list:
                new_segments = []
                for seg in text_segments:
                    if seg not in self.unique_no_split_tokens:
                        new_segments.extend(split_on_token(tok, seg))
                    else:
                        new_segments.append(seg)
                text_segments = new_segments
            return list(
                itertools.chain.from_iterable(
                    (
                        self._basic_tokenize(tok) if tok not in self.unique_no_split_tokens else [tok]
                        for tok in text_segments
                    )
                )
            )

        return split_on_tokens(self.unique_no_split_tokens, text)

    def encode(self, text: str) -> List[int]:
        return [self.get_idx(tok) for tok in self.tokenize(text)]

    def get_batch_converter(self, truncation_seq_length: int = None):
        if self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        return BatchConverter(self, truncation_seq_length)


class BatchConverter:
    def __init__(self, alphabet: Alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        batch_size = len(raw_batch)
        labels, seq_strs = zip(*raw_batch)
        seq_encoded = [self.alphabet.encode(s) for s in seq_strs]

        if self.truncation_seq_length:
            seq_encoded = [s[: self.truncation_seq_length] for s in seq_encoded]

        max_len = max(len(s) for s in seq_encoded)
        tokens = paddle.full(
            shape=[batch_size, max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)],
            fill_value=self.alphabet.padding_idx,
            dtype="int64",
        )

        for i, seq_ids in enumerate(seq_encoded):
            offset = 0
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
                offset = 1
            tokens[i, offset : offset + len(seq_ids)] = paddle.to_tensor(seq_ids, dtype="int64")
            if self.alphabet.append_eos:
                tokens[i, offset + len(seq_ids)] = self.alphabet.eos_idx

        return list(labels), list(seq_strs), tokens


class MSABatchConverter(BatchConverter):
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            raw_batch: Sequence[RawMSA] = [inputs]
        else:
            raw_batch = inputs

        batch_size = len(raw_batch)
        max_align = max(len(msa) for msa in raw_batch)
        max_len = max(len(msa[0][1]) for msa in raw_batch)

        tokens = paddle.full(
            shape=[
                batch_size,
                max_align,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ],
            fill_value=self.alphabet.padding_idx,
            dtype="int64",
        )

        batch_labels, batch_strs = [], []

        for i, msa in enumerate(raw_batch):
            labels, strs, msa_tokens = super().__call__(msa)
            batch_labels.append(labels)
            batch_strs.append(strs)
            tokens[i, : msa_tokens.shape[0], : msa_tokens.shape[1]] = msa_tokens

        return batch_labels, batch_strs, tokens
