# step 1) sequence masking
from .Function import train, generate
from .StructPred import predict_pdbs, predict_pdbs_fast_no_msa
import pandas as pd

def PEP_Augmentation(PEPid, seed=42):
    pep_dir = 'Data/pos_label.csv'
    pdb_dir = 'Data/3D_structure'
    pep_df= pd.read_csv(pep_dir)
    pep_selected = pep_df[pep_df['Peptides'].astype(str).isin(PEPid)].drop_duplicates('Peptides')
    if pep_selected.empty:
        raise RuntimeError("No matches were found in CSV for the provided PEP IDs.")

    #step 1) pretraining（alpha=0.0 not using contact map)
    train(df=pep_selected,
          structure=pdb_dir,
          epochs=300,
          batch_size=128,
          n_neighbors=5,
          d_model=128,
          nhead=4,
          layers=3,
          lr=1e-4,
          alpha=1.0,
          ckpt='PEP_augment/checkpoint/pep_aug.pt',
          seed=seed,
          min_len=5,
          max_len=80)
    
    #step 2) generating peptide samples
    generate(df=pep_selected,
             checkpoint='PEP_augment/checkpoint/pep_aug.pt',
             k=5, 
             n_neighbors=5,
             seed=seed,
             out='PEP_augment/data_augs/PEP_augs.csv',
             min_len=1,
             max_len=100)
    
    # #step ?)  predicting peptide augment structures
    # predict_pdbs_fast_no_msa("./data_augs/CV_Independent_merged.csv", 
    #                          "./data_augs/3D_augs",
    #                          id_col="Peptide", 
    #                          seq_col="Sequence",
    #                          overwrite=True)
    # predict_pdbs("data_augs/PEP_augs.csv", "data_augs/3D_augs",
    #              id_col="variant_id", seq_col="Sequence", overwrite=True)
    
    return 0


