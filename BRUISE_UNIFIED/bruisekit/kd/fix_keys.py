#!/usr/bin/env python3
"""
fix_keys.py — convert new-transformers SegFormer weights to the server's classic
naming, so the Colab-made B2/B0 checkpoints + mit_b0/mit_b2 load in the older
transformers on mlidl.

New (Colab)                                   ->  Classic (server)
  segformer.stages.{S}.patch_embeddings.       ->  segformer.encoder.patch_embeddings.{S}.
  segformer.stages.{S}.blocks.{B}.             ->  segformer.encoder.block.{S}.{B}.
  segformer.stages.{S}.layer_norm.             ->  segformer.encoder.layer_norm.{S}.
  ...layernorm_before / layernorm_after        ->  ...layer_norm_1 / layer_norm_2
  ...attention.q_proj/k_proj/v_proj            ->  ...attention.self.query/key/value
  ...attention.o_proj                          ->  ...attention.output.dense
  ...attention.sequence_reduction.sequence_reduction -> ...attention.self.sr
  ...attention.sequence_reduction.layer_norm   ->  ...attention.self.layer_norm
  ...mlp.fc1 / mlp.fc2                          ->  ...mlp.dense1 / mlp.dense2
  decode_head.linear_projections.{i}           ->  decode_head.linear_c.{i}
(mlp.dwconv.dwconv, linear_fuse, batch_norm, classifier are already identical.)

Usage (on the server, in the bruise env, from ~/segformer_b5_bruise_pipeline):
    python fix_keys.py

It fixes, in place (with .bak backups):
    teachers/segformer_b2_teacher/best_model.pt
    teachers/segformer_b0_distilled/best_model.pt
    pretrained_weights/segformer_mit_b0/model.safetensors
    pretrained_weights/segformer_mit_b2/model.safetensors
and STRICT-load-tests the two teacher checkpoints against a classic-built model.
Idempotent: already-classic files are detected and skipped.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch


def remap_key(k: str) -> str:
    # structural: stages.{S}.blocks.{B}. -> encoder.block.{S}.{B}.
    k = re.sub(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.",
               r"segformer.encoder.block.\1.\2.", k)
    # stages.{S}.patch_embeddings. -> encoder.patch_embeddings.{S}.
    k = re.sub(r"segformer\.stages\.(\d+)\.patch_embeddings\.",
               r"segformer.encoder.patch_embeddings.\1.", k)
    # stages.{S}.layer_norm. -> encoder.layer_norm.{S}.   (stage-level norm only)
    k = re.sub(r"segformer\.stages\.(\d+)\.layer_norm\.",
               r"segformer.encoder.layer_norm.\1.", k)
    # within-block renames
    k = k.replace("layernorm_before", "layer_norm_1")
    k = k.replace("layernorm_after", "layer_norm_2")
    k = k.replace("attention.sequence_reduction.sequence_reduction", "attention.self.sr")
    k = k.replace("attention.sequence_reduction.layer_norm", "attention.self.layer_norm")
    k = k.replace("attention.q_proj", "attention.self.query")
    k = k.replace("attention.k_proj", "attention.self.key")
    k = k.replace("attention.v_proj", "attention.self.value")
    k = k.replace("attention.o_proj", "attention.output.dense")
    k = k.replace("mlp.fc1", "mlp.dense1")
    k = k.replace("mlp.fc2", "mlp.dense2")
    k = k.replace("decode_head.linear_projections.", "decode_head.linear_c.")
    return k


def is_new_format(keys) -> bool:
    return any(".stages." in k or "q_proj" in k or "linear_projections" in k for k in keys)


def remap_state_dict(sd: dict) -> dict:
    return {remap_key(k): v for k, v in sd.items()}


# --- checkpoints (.pt, SegformerWrapper state dicts) ---------------------------------
def fix_checkpoint(path: Path):
    if not path.exists():
        print(f"  [skip] not found: {path}"); return
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if not is_new_format(sd.keys()):
        print(f"  [ok] already classic: {path}"); return
    shutil.copy(str(path), str(path) + ".bak")
    new = remap_state_dict(sd)
    torch.save(new, str(path))
    print(f"  [fixed] {path}  ({len(new)} keys; backup {path}.bak)")


# --- pretrained encoders (model.safetensors) -----------------------------------------
def fix_safetensors(dir_path: Path):
    st = dir_path / "model.safetensors"
    if not st.exists():
        print(f"  [skip] not found: {st}"); return
    from safetensors.torch import load_file, save_file
    sd = load_file(str(st))
    if not is_new_format(sd.keys()):
        print(f"  [ok] already classic: {st}"); return
    shutil.copy(str(st), str(st) + ".bak")
    new = remap_state_dict(sd)
    save_file(new, str(st), metadata={"format": "pt"})
    print(f"  [fixed] {st}  ({len(new)} keys; backup {st}.bak)")


# --- strict-load self-test of a teacher checkpoint -----------------------------------
def strict_test(ckpt: Path, pretrained: Path):
    if not ckpt.exists():
        return
    from transformers import SegformerForSemanticSegmentation
    import torch.nn as nn, torch.nn.functional as F

    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.model = m
    hf = SegformerForSemanticSegmentation.from_pretrained(
        str(pretrained), num_labels=1, ignore_mismatched_sizes=True)
    model = W(hf)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("model.mean") and not m.startswith("model.std")]
    if missing or unexpected:
        print(f"  [FAIL] {ckpt}\n     missing={missing[:4]}...\n     unexpected={unexpected[:4]}...")
        raise SystemExit("key remap incomplete — do not proceed; paste this output back.")
    print(f"  [PASS strict load] {ckpt}")


def main():
    root = Path(".")
    print("== fixing checkpoints ==")
    fix_checkpoint(root / "teachers/segformer_b2_teacher/best_model.pt")
    fix_checkpoint(root / "teachers/segformer_b0_distilled/best_model.pt")
    print("== fixing pretrained encoders ==")
    fix_safetensors(root / "pretrained_weights/segformer_mit_b0")
    fix_safetensors(root / "pretrained_weights/segformer_mit_b2")
    print("== strict-load self-test ==")
    strict_test(root / "teachers/segformer_b2_teacher/best_model.pt",
                root / "pretrained_weights/segformer_mit_b2")
    strict_test(root / "teachers/segformer_b0_distilled/best_model.pt",
                root / "pretrained_weights/segformer_mit_b0")
    print("\n[done] all SegFormer weights are now in the server's classic naming.")


if __name__ == "__main__":
    main()
