#!/usr/bin/env python3
import os
import pickle 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from Utils import datacompile, esm2_feature, PairDataset, evaluate_clf
from PseudoLabel import mean_teacher_train, pseudo_label_with_teacher
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from PEP_augment.Pretrain import PEP_Augmentation
from Model import MLP
import random

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

rd = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
need_aug = False

dataset_df = datacompile("./Data/", neg_fold=5, seed=rd)
pepid = list(dataset_df.index)
seq_feat = esm2_feature(list(zip(dataset_df.index, dataset_df['Sequence'])))
seq_feat = pd.DataFrame(seq_feat, index=dataset_df.index)

CV, Independent = train_test_split(dataset_df, test_size=0.2, random_state=rd)

if need_aug:
    PEP_Augmentation(PEPid=set(CV[CV.iloc[:, 1]==1].index), seed=rd)

PEP_A = pd.read_csv('PEP_augment/data_augs/PEP_augs.csv', index_col=0)
pepid_A = list(PEP_A['variant_id'])
seq_feat_A = esm2_feature(list(zip(PEP_A['variant_id'], PEP_A['Sequence'])))
seq_feat_A = pd.DataFrame(seq_feat_A, index=pepid_A)

#%%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rd)
X_df = CV.index.values
Y_df = CV['Label'].values
fold_metrics = []
ind_set = PairDataset(Independent, seq_feat)
ind_loader = DataLoader(ind_set, batch_size=128, shuffle=False)

for k, (tr_idx, va_idx) in enumerate(kf.split(X_df, Y_df), start=1):
    print(f"\n===== Fold {k}/5 =====")
    
    # === 每个 fold 重新固定随机种子（关键稳定措施）===
    seed_everything(rd + k)
    
    tr_df = CV.iloc[tr_idx]
    va_df = CV.iloc[va_idx]

    train_set = PairDataset(tr_df, seq_feat)
    val_set   = PairDataset(va_df, seq_feat)
    tr_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    va_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    
    # === 模型改小一点 + dropout 提高（防止过拟合）===
    model_T = MLP(seq_feat.shape[1], hidden=(512,128,32), p_drop=0.2)
    teacher_save = os.path.join('Results', f'fold{k}_teacher.pt')
    
    model_T, teacher_ema = mean_teacher_train(
        model_T, tr_loader, va_loader,
        epochs=300, lr=1e-4, alpha=0.99, patience=20,
        save_path=teacher_save
    )
    
    pseudo_df = pseudo_label_with_teacher(
        teacher_ema, seq_feat_A, list(PEP_A['Sequence']),
        tau_pos=0.90, tau_neg=0.10, temperature=2.0
    )
    
    stu_train_df = pd.concat([tr_df, pseudo_df])
    stu_train_df['Label'] = stu_train_df['Label'].astype('float32')
    seq_feat_all = pd.concat([seq_feat, seq_feat_A])
    stu_set = PairDataset(stu_train_df, seq_feat_all)
    stu_tr = DataLoader(stu_set, batch_size=256, shuffle=True)
    
    model_S = MLP(seq_feat.shape[1], hidden=(512,128,32), p_drop=0.2)
    student_save = os.path.join('Results', f'fold{k}_student.pt')
    model_S, _ = mean_teacher_train(
        model_S, stu_tr, va_loader,
        epochs=300, lr=1e-4, alpha=0.99, patience=20,
        save_path=student_save
    )
    
    m = model_S.to(device)
    val_metrics = evaluate_clf(m, ind_loader)
    print('================indepent testing=================')
    print({k: f"{float(val_metrics[k]):.4f}" for k in ['auroc','auprc','acc','f1']})
    fold_metrics.append(val_metrics)

mean = {k: float(np.mean([d[k] for d in fold_metrics])) for k in ['auroc','auprc','acc','f1']}
print(mean)  

