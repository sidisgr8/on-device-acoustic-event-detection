"""Global magnitude (L1) unstructured pruning with masked fine-tuning (spec 4.6, D-31).

* Prunable tensors: the weights of every Conv2d, Linear and GRU (weight_ih_*/weight_hh_*) layer. Biases,
  BatchNorm/LayerNorm parameters, embeddings (position, cls, distillation) are never pruned. The classifier
  head is pruned like any other layer (one global threshold).
* One global threshold on |w| across all prunable tensors zeroes exactly the requested fraction.
* Masks are explicit and re-applied after every optimiser step, so pruned weights stay exactly zero.
  (torch.nn.utils.prune is not used: it does not cope well with GRU flat weights.)
* Unstructured sparsity shrinks only the *compressed* file, not CPU latency: the weights stay dense in memory.
"""
import copy
import gzip
import io

import torch
from torch import nn

NOTE = ("Unstructured sparsity does not reduce CPU latency by itself: the pruned tensors stay dense "
        "(zeros are still multiplied). Only the gzip-compressed size and the non-zero parameter count shrink.")


def prunable_tensors(model: nn.Module) -> dict:
    out = {}
    for mname, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            out[f"{mname}.weight"] = m.weight
        elif isinstance(m, nn.GRU):
            for pname, p in m.named_parameters(recurse=False):
                if pname.startswith("weight_"):
                    out[f"{mname}.{pname}"] = p
    return out


def global_magnitude_masks(model: nn.Module, sparsity: float) -> dict:
    """Boolean keep-masks (True = keep) that zero the `sparsity` fraction of smallest |w| globally."""
    tensors = prunable_tensors(model)
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(sparsity)
    allw = torch.cat([t.detach().abs().flatten().float().cpu() for t in tensors.values()])
    k = int(round(sparsity * allw.numel()))
    if k == 0:
        return {n: torch.ones_like(t, dtype=torch.bool) for n, t in tensors.items()}
    thr = torch.kthvalue(allw, k).values.item()
    masks = {n: t.detach().abs().float().cpu() > thr for n, t in tensors.items()}
    return masks


@torch.no_grad()
def apply_masks(model: nn.Module, masks: dict):
    tensors = prunable_tensors(model)
    for n, m in masks.items():
        tensors[n].mul_(m.to(tensors[n].device, tensors[n].dtype))


def sparsity_report(model: nn.Module) -> dict:
    tensors = prunable_tensors(model)
    prunable = sum(t.numel() for t in tensors.values())
    nz_prunable = sum(int((t != 0).sum()) for t in tensors.values())
    total = sum(p.numel() for p in model.parameters())
    nz_total = sum(int((p != 0).sum()) for p in model.parameters())
    return {"total_params": total, "nonzero_params": nz_total, "prunable_params": prunable,
            "nonzero_prunable": nz_prunable, "actual_sparsity_of_prunable": 1 - nz_prunable / prunable,
            "overall_nonzero_fraction": nz_total / total}


def sizes(model: nn.Module) -> dict:
    """Dense fp32 file size and gzip(level 9) size of the saved state_dict."""
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    raw = buf.getvalue()
    return {"dense_fp32_bytes": len(raw), "gzip_bytes": len(gzip.compress(raw, compresslevel=9))}


def finetune_lr(base_lr: float) -> float:
    """D-31: fine-tune at 0.1 x the original training learning rate. Fixed in advance, not tuned."""
    return 0.1 * base_lr
