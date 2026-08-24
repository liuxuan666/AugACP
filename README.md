# AugACP
Source code and data for "Improving Anticancer Peptide Prediction via A Semi-Supervised Learning Framework with Bio-inspired Augmentation."

![Framework of AugACP](https://github.com/liuxuan666/AugACP/blob/main/model.png)  

# Requirements
* Python >= 3.10
* PyTorch >= 2.6
* fair-esm >= 1.1


# Usage
* First, please extract the file "Data/3D_structure.zip" from the original directory. It should be noted that **“Data/neg_label.csv”** contains the complete set of candidate negative samples collected from the Human PeptideAtlas database. During data preparation, our code randomly samples negative samples at a **5:1 negative-to-positive ratio** from this candidate pool.

* Next, the following scenarios can be tested:
* python Main.py \<parameters\>  #---Binary classification task with 5-fold CV

* "Data/EV.csv" is the dataset used for external validation, and "Data/EV_predictions.csv" is the prediction results of the external validation set.
