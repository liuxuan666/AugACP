import esm
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def datacompile(path, neg_fold, seed=42):
    # 读取pos_label.csv
    path = "./Data/"
    pos_df = pd.read_csv(path + 'pos_label.csv')
    pos_ids = pos_df['Peptides'].tolist()
    pos_sequences = pos_df['Sequence'].tolist()
    # 读取neg_label.csv
    neg_df = pd.read_csv(path + 'neg_label.csv')
    neg_ids = neg_df['Peptides'].tolist()
    neg_sequences = neg_df['Sequence'].tolist()
    
    len_pos = len(pos_sequences)
    random.seed(seed)
    # 创建负样本的ID和Sequence对
    neg_pairs = list(zip(neg_ids, neg_sequences))
    neg_sampled_pairs = random.sample(neg_pairs, min(neg_fold * len_pos, len(neg_pairs)))
    neg_sampled_ids, neg_sampled_sequences = zip(*neg_sampled_pairs)
    # 创建数据集
    data = []
    for pid, seq in zip(pos_ids, pos_sequences):
        data.append({'Peptide': pid, 'Sequence': seq, 'Label': 1})
    for pid, seq in zip(neg_sampled_ids, neg_sampled_sequences):
        data.append({'Peptide': pid, 'Sequence': seq, 'Label': 0})
    # 转换为DataFrame并打乱
    dataset_df = pd.DataFrame(data)
    dataset_df = dataset_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # 设置Peptide作为index
    dataset_df = dataset_df.set_index('Peptide')
     
    return dataset_df


def esm2_feature(data, batch_size=64, layer=33):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    # Generate per-sequence representations via averaging
    seq_reps = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            _, _, batch_tokens = batch_converter(chunk)
            batch_tokens = batch_tokens.to(device)
            out = model(batch_tokens, repr_layers=[layer], return_contacts=False)
            tok_repr = out["representations"][layer]  # [B, T, H]
            lens = (batch_tokens != alphabet.padding_idx).sum(1)  # [B]
            for b in range(tok_repr.size(0)):
                L = int(lens[b].item())
                start, end = 1, max(1, L - 1)
                toks = tok_repr[b, start:end].mean(0)
                seq_reps.append(toks.detach().cpu().numpy())
            torch.cuda.empty_cache()
    
    return np.stack(seq_reps, axis=0)

class PairDataset(Dataset):
    def __init__(self, df_pairs: pd.DataFrame,
                 pep_seq_df: pd.DataFrame,
                 strict: bool = True):
        
        self.pep_ids = df_pairs.index.astype(str).tolist()
        self.labels = torch.tensor(df_pairs['Label'].values, dtype=torch.float32)    
        self.P1 = torch.tensor(pep_seq_df.loc[self.pep_ids].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.pep_ids)
    
    def __getitem__(self, i):
        if self.labels is None:
            return self.P1[i]
        return self.P1[i], self.labels[i]

@torch.no_grad()
def evaluate_clf(model, loader):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        p1, y= batch
        p1 = p1.to(device)
        logit = model(p1)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(prob)
        if y is not None:
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    out = {'probs': probs}
    if labels:
        y_true = np.concatenate(labels)
        out.update({
            'auroc': float(roc_auc_score(y_true, probs)),
            'auprc': float(average_precision_score(y_true, probs)),
            'acc':   float(accuracy_score(y_true, (probs>=0.5).astype(int))),
            'f1':    float(f1_score(y_true, (probs>=0.5).astype(int)))
        })
    return out