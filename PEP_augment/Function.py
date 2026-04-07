import os
import math
import numpy as np
import pandas as pd
import torch
from .Support import build_masked_samples, to_ids, to_seq, PAD_ID, MASK_ID, ITOA

# ---- Model ----
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_ff), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_ff, d_model)
        )
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x, pad_mask=None):
        kpm = (pad_mask == 0) if pad_mask is not None else None
        y, attn = self.mha(x, x, x, key_padding_mask=kpm, need_weights=True, average_attn_weights=False)
        attn = attn.mean(1)  # [B,H,L,L] -> [B,L,L]
        x = self.ln1(x + self.drop(y))
        y = self.ff(x)
        x = self.ln2(x + self.drop(y))
        return x, attn

class TransformerMLM(torch.nn.Module):
    def __init__(self, vocab_size=22, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.tok = torch.nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = torch.nn.ModuleList([EncoderLayer(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.lm = torch.nn.Linear(d_model, vocab_size)
    def forward(self, input_ids, pad_mask=None):
        x = self.tok(input_ids)
        x = self.pos(x)
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, pad_mask)
            attn_list.append(attn)
        logits = self.lm(x)
        A = torch.stack(attn_list, dim=0).mean(0)
        return logits, A

# ---- Optional contact map (import inside function; no checks) ----
def contact_map_from_pdb(pdb_path, cutoff=8.0):
    from Bio.PDB import PDBParser  # import only if you actually call this
    import numpy as np
    if not os.path.exists(pdb_path): 
        return None
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('x', pdb_path)
    atoms = []
    for model in structure:
        for chain in model:
            for res in chain:
                if 'CA' in res: atoms.append(res['CA'].get_coord())
                elif 'N' in res: atoms.append(res['N'].get_coord())
            break
        break
    if not atoms: return None
    coords = np.stack(atoms)
    d = np.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=-1)
    C = (d < cutoff).astype(np.float32)
    np.fill_diagonal(C, 0.0)
    return C


def train(df, structure, epochs, batch_size, n_neighbors, d_model, nhead, layers,
          lr, alpha, ckpt, seed, min_len, max_len):
    torch.manual_seed(seed)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    samples = build_masked_samples(df, n_neighbors, 2, 3, seed, min_len, max_len)

    if len(samples) == 0:
        raise RuntimeError(f'No sequences within length range [{min_len}, {max_len}] found. Nothing to train.')

    X, Y, M, LENS, IDS = [], [], [], [], []
    for pid, orig, masked, mask_idx in samples:
        x = to_ids(masked); y = to_ids(orig)
        tgt = np.full(len(y), -100, dtype=np.int64)
        for mi in mask_idx: tgt[mi] = y[mi]
        X.append(x); Y.append(tgt); M.append([1]*len(x)); LENS.append(len(x)); IDS.append(pid)

    maxlen = max(LENS) if LENS else 1
    def pad(arr, val): return np.array(arr + [val]*(maxlen-len(arr)), dtype=np.int64)
    X = np.stack([pad(x, PAD_ID) for x in X])
    Y = np.stack([pad(y.tolist(), -100) for y in Y])
    M = np.stack([pad(m, 0) for m in M])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerMLM(vocab_size=len(ITOA), d_model=d_model, nhead=nhead, num_layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def batch_contact(ids, L_pad):
        batch = []
        for pid in ids:
            p = os.path.join(structure, f'{pid}.pdb')
            C = contact_map_from_pdb(p)
            if C is None:
                return None
            C = np.asarray(C, dtype=np.float32)
            n = C.shape[0]
            if n < L_pad:
                padm = np.zeros((L_pad, L_pad), dtype=np.float32)
                padm[:n, :n] = C
                C = padm
            elif n > L_pad:
                C = C[:L_pad, :L_pad]
            batch.append(C)
        return torch.tensor(np.stack(batch), dtype=torch.float32, device=device)


    for ep in range(1, epochs+1):
        model.train(); total=0.0
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.long, device=device)
            yb = torch.tensor(Y[i:i+batch_size], dtype=torch.long, device=device)
            mb = torch.tensor(M[i:i+batch_size], dtype=torch.long, device=device)
            logits, A = model(xb, mb)
            loss = ce(logits.view(-1, logits.size(-1)), yb.view(-1))
            if alpha > 0.0:
                ids_b = IDS[i:i+batch_size]
                L_pad = mb.size(1)
                C = batch_contact(ids_b, L_pad)
                if C is not None:
                    valid = (mb>0).unsqueeze(1) & (mb>0).unsqueeze(2)
                    attn = torch.sigmoid(A) * valid
                    contact = C * valid
                    loss = loss + alpha * torch.nn.functional.mse_loss(attn, contact)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        print(f"Epoch {ep}: loss={total/max(1, math.ceil(len(X)/batch_size)):.4f}")

    torch.save({'model': model.state_dict(), 'meta': {'d_model': d_model, 'nhead': nhead, 'layers': layers}}, ckpt)
    print(f"Saved checkpoint to {ckpt}")


def generate(df, checkpoint, k, n_neighbors, seed, out, min_len, max_len):
    torch.manual_seed(seed)
    samples = build_masked_samples(df, n_neighbors, 2, 3, seed, min_len, max_len)
    if len(samples) == 0:
        raise RuntimeError(f'No sequences within length range [{min_len}, {max_len}] found. Nothing to generate.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck = torch.load(checkpoint, map_location=device)
    meta = ck.get('meta', {})
    model = TransformerMLM(vocab_size=len(ITOA), d_model=meta.get('d_model',128), nhead=meta.get('nhead',4), num_layers=meta.get('layers',3)).to(device)
    model.load_state_dict(ck['model']); model.eval()

    rows = []
    with torch.no_grad():
        for pid, orig, masked, _ in samples:
            ids = torch.tensor([to_ids(masked)], dtype=torch.long, device=device)
            mask = (ids != PAD_ID).long()
            logits, _ = model(ids, mask)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            seq_ids = ids[0].cpu().tolist()
            mask_positions = [i for i,tok in enumerate(seq_ids) if tok == MASK_ID]
            if not mask_positions:
                rows.append({'Peptides': pid, 'variant_id': f'{pid}_var1', 'Sequence': to_seq(seq_ids)})
                continue
            topk_preds = [np.argsort(-probs[i])[:k] for i in mask_positions]
            from itertools import product
            combos = list(product(*topk_preds))[:k]
            for ci, combo in enumerate(combos, 1):
                new_ids = seq_ids[:]
                for pos, pred in zip(mask_positions, combo):
                    new_ids[pos] = int(pred)
                rows.append({'Peptides': pid, 'variant_id': f'{pid}_var{ci}', 'Sequence': to_seq(new_ids)})
    out_dir = os.path.dirname(out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} variants -> {out}")
