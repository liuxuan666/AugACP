import os
import pandas as pd
import torch
import torch.nn as nn
import copy
from Utils import evaluate_clf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Mean Teacher 主训练函数（稳定性优化） ======================
def mean_teacher_train(model, train_loader, val_loader, epochs=300, lr=1e-4, 
                       alpha=0.995, patience=30, save_path=None):
    student = model.to(device)
    teacher = copy.deepcopy(student)
    teacher.eval()
    
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    best, best_state, noimp = -1.0, None, 0

    for ep in range(1, epochs + 1):
        student.train()
        for p1, y in train_loader:
            p1 = p1.to(device)
            y = y.to(device).float()
            
            logit = student(p1)
            loss = bce(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # EMA 更新 teacher
            with torch.no_grad():
                for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                    t_param.data.mul_(alpha).add_((1 - alpha) * s_param.data)

        val = evaluate_clf(student, val_loader)
        if val.get('auroc', -1) > best:
            best = val['auroc']
            best_state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
            noimp = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
        else:
            noimp += 1
        if noimp >= patience:
            break

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, teacher


# ====================== Soft Pseudo-Label ======================
@torch.no_grad()
def pseudo_label_with_teacher(teacher: nn.Module,                            
                              pep_seq_A: pd.DataFrame, 
                              seqname,
                              tau_pos=0.82,      
                              tau_neg=0.18,
                              temperature=2.5) -> pd.DataFrame:  
    teacher = teacher.to(device).eval()
    kept_rows = []
    
    P1 = torch.tensor(pep_seq_A.values, dtype=torch.float32, device=device)
    logit = teacher(P1)
    prob = torch.sigmoid(logit / temperature).detach().cpu().numpy()
    
    for (z, n, p) in zip(pep_seq_A.index, seqname, prob):
        if p >= tau_pos:
            kept_rows.append((z, n, float(p)))
        elif p <= tau_neg:
            kept_rows.append((z, n, 0.0))
            
    df = pd.DataFrame(kept_rows, columns=['Peptide', 'Sequence', 'Label'])
    df.set_index('Peptide', inplace=True)
    return df