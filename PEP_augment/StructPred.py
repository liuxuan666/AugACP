from __future__ import annotations
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import shutil

DEFAULT_LOCALCOLABFOLD = "/home/lx/Project/localcolabfold/localcolabfold/colabfold-conda/bin/colabfold_batch"

def _resolve_colabfold_exe() -> str:
    exe = os.environ.get("COLABFOLD_BIN", "").strip()
    if exe and Path(exe).exists():
        return exe
    if Path(DEFAULT_LOCALCOLABFOLD).exists():
        return DEFAULT_LOCALCOLABFOLD
    from shutil import which
    exe = which("colabfold_batch")


_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
_NON_LET = re.compile(r"[^A-Za-z]")
_SAFE_ID = re.compile(r'[\\/:*?"<>|\s]+')

def _clean_seq(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _NON_LET.sub("", s).upper()
    if not s:
        return ""
    return "".join(ch if ch in _VALID_AA else "G" for ch in s)

def _pick_cols(df: pd.DataFrame, id_col: Optional[str], seq_col: Optional[str]) -> Tuple[str, str]:
    cand_id  = [c for c in [id_col, "variant_id", "Peptides", "ID", "Name", "id", "name"] if c and c in df.columns]
    cand_seq = [c for c in [seq_col, "Sequence", "Peptides", "sequence", "seq", "AA"] if c and c in df.columns]
    if not cand_id:
        df["__ROW_ID__"] = [f"pep_{i}" for i in range(len(df))]
        cand_id = ["__ROW_ID__"]
    if not cand_seq:
        raise ValueError("Cannot locate sequence column (tried: Sequence/Peptides/sequence/seq/AA)")
    return cand_id[0], cand_seq[0]

def _write_fasta(pairs: List[Tuple[str, str]], path: Path) -> None:
    with open(path, "w") as f:
        for rid, seq in pairs:
            f.write(f">{rid}\n{seq}\n")

def _pick_rank1(seq_dir: Path) -> Optional[Path]:
    # rank_001
    hits = list(seq_dir.rglob("*rank_001*.pdb"))
    if hits:
        return hits[0]
    hits = list(seq_dir.rglob("*.pdb"))
    return hits[0] if hits else None

def _gather(tmp_out: Path, out_dir: Path, ids: List[str]) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {p.name: p for p in tmp_out.iterdir() if p.is_dir()}
    ok = miss = 0
    for rid in ids:
        dst = out_dir / f"{rid}.pdb"
        if dst.exists():
            ok += 1
            continue
        hit = None
        if rid in subdirs:
            hit = _pick_rank1(subdirs[rid])
        if hit is None:
            cand = list(tmp_out.rglob(f"*{rid}*rank_001*.pdb")) or list(tmp_out.rglob(f"*{rid}*.pdb"))
            if cand:
                hit = cand[0]
        if hit and hit.exists():
            dst.write_bytes(hit.read_bytes())
            ok += 1
        else:
            print(f"[StructPred] WARN no PDB for {rid}")
            miss += 1
    return ok, miss

# ==========main==========
def predict_pdbs(csv_path: str,
                 out_dir: str,
                 id_col: str = "variant_id",
                 seq_col: str = "Sequence",
                 overwrite: bool = False) -> None:
   
    exe = _resolve_colabfold_exe()
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    id_col, seq_col = _pick_cols(df, id_col, seq_col)

    pairs: List[Tuple[str, str]] = []
    for i, r in df.iterrows():
        rid_raw = str(r[id_col]) if pd.notna(r[id_col]) else f"pep_{i}"
        rid = _SAFE_ID.sub("_", rid_raw).strip("_") or f"pep_{i}"
        seq = _clean_seq(r[seq_col])
        if not seq:
            continue
        if (not overwrite) and (outp / f"{rid}.pdb").exists():
            continue
        pairs.append((rid, seq))

    if not pairs:
        print("[StructPred] nothing to do."); return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "batch.fasta"
        tmp_out = td / "cf_out"
        _write_fasta(pairs, fasta)
        tmp_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            exe,
            str(fasta),
            str(tmp_out),
            "--num-models", "1",
            "--num-recycle", "3",
            "--stop-at-score", "0.75",
            # "--disable-amber",
        ]
        print("[StructPred] cmd:\n ", " ".join(cmd))
        env = os.environ.copy()
        env.pop("MPLBACKEND", None)
        env["MPLBACKEND"] = "Agg"

        subprocess.run(cmd, check=True, env=env)

        ids = [rid for rid, _ in pairs]
        ok, miss = _gather(tmp_out, outp, ids)
        print(f"[StructPred] OK:{ok} missing:{miss} -> {out_dir}")


def predict_pdbs_fast_no_msa(csv_path: str,
                             out_dir: str,
                             id_col: str = "variant_id",
                             seq_col: str = "Sequence",
                             overwrite: bool = False):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    pairs: List[Tuple[str, str]] = []
    for i, r in df.iterrows():
        rid_raw = str(r[id_col]) if pd.notna(r[id_col]) else f"pep_{i}"
        rid = _SAFE_ID.sub("_", rid_raw).strip("_") or f"pep_{i}"
        seq = _clean_seq(r[seq_col])
        if not seq:
            continue
        if (not overwrite) and (outp / f"{rid}.pdb").exists():
            continue
        pairs.append((rid, seq))

    if not pairs:
        print("[StructPred] nothing to do.")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "batch.fasta"
        tmp_out = td / "cf_out"
        _write_fasta(pairs, fasta)
        tmp_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "colabfold_batch", str(fasta), str(tmp_out),
            "--msa-mode", "single_sequence",
            "--num-models", "1",
            "--num-recycle", "3",
            "--stop-at-score", "0.75",
            # "--disable-amber",  # 若不需要Relax可解开进一步加速
        ]
        print("[StructPred] run:\n ", " ".join(cmd))
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        subprocess.run(cmd, check=True, env=env)

        # 仅保留 .pdb：从 tmp_out 递归查找并复制到目标目录
        kept = 0
        for p in tmp_out.rglob("*.pdb"):
            target = outp / p.name
            # 若你希望强制用表格ID命名，可在此处重命名；否则保持 colabfold 的 ranked 命名
            shutil.copy2(p, target)
            kept += 1

        print(f"[StructPred] kept {kept} PDBs -> {out_dir}")

