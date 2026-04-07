import random
import pandas as pd

# ---- Vocab ----
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL = ["[PAD]", "[MASK]"]
ITOA = {i: aa for i, aa in enumerate(SPECIAL + AMINO_ACIDS)}
ATOI = {aa: i for i, aa in ITOA.items()}
PAD_ID = ATOI["[PAD]"]
MASK_ID = ATOI["[MASK]"]

def to_ids(seq: str):
    out = []
    i = 0
    while i < len(seq):
        if seq.startswith("[MASK]", i):
            out.append(MASK_ID); i += len("[MASK]")
        else:
            out.append(ATOI.get(seq[i], PAD_ID)); i += 1
    return out

def to_seq(ids):
    return "".join(ITOA.get(i, "") for i in ids if i >= 2)

# ---- String metrics ----
def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[lb]

def lcs_with_indices(a: str, b: str):
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la-1, -1, -1):
        for j in range(lb-1, -1, -1):
            dp[i][j] = 1 + dp[i+1][j+1] if a[i]==b[j] else max(dp[i+1][j], dp[i][j+1])
    i=j=0; idxs=[]; out=[]
    while i<la and j<lb:
        if a[i]==b[j]:
            out.append(a[i]); idxs.append(i); i+=1; j+=1
        elif dp[i+1][j] >= dp[i][j+1]:
            i+=1
        else:
            j+=1
    return "".join(out), idxs

# ---- Augmentation ----
PRESERVE = set(list("KRHLG"))

def top_n_neighbors(i, seqs, n):
    base = seqs[i]
    d = sorted(((j, levenshtein(base, s)) for j,s in enumerate(seqs) if j!=i), key=lambda x:x[1])
    return [j for j,_ in d[:n]]

def conserved_positions(i, seqs, n, vote_thresh=None):
    base = seqs[i]
    neigh = top_n_neighbors(i, seqs, n)
    votes = [0]*len(base)
    for j in neigh:
        _, idxs = lcs_with_indices(base, seqs[j])
        for k in idxs: votes[k]+=1
    if vote_thresh is None: vote_thresh = max(1, len(neigh)//2)
    conserved = {idx for idx,v in enumerate(votes) if v>=vote_thresh}
    conserved |= {idx for idx,ch in enumerate(base) if ch in PRESERVE}
    return conserved, set(neigh)

def mask_non_conserved(seq, conserved, mask_min=2, mask_max=3):
    nonc = [i for i in range(len(seq)) if i not in conserved]
    if not nonc: return seq, []
    m = min(random.randint(mask_min, mask_max), len(nonc))
    picks = random.sample(nonc, m)
    s = list(seq)
    for i in picks: s[i] = "[MASK]"
    return "".join(s), picks

def build_masked_samples(df, n_neighbors, mask_min, mask_max, seed, min_len, max_len):
    random.seed(seed)
    ids = df['Peptides'].tolist()
    seqs = df['Sequence'].tolist()
    samples = []
    for i, seq in enumerate(seqs):
        L = len(seq)
        if not (min_len <= L <= max_len):
            continue  # skip sequences out of [min_len, max_len]
        conserved, _ = conserved_positions(i, seqs, n_neighbors)
        masked, mask_idx = mask_non_conserved(seq, conserved, mask_min, mask_max)
        samples.append((ids[i], seq, masked, mask_idx))
    return samples
