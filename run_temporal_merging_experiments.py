"""
Unified temporal merging experiment runner.

Runs 3 experiments in one script:
  1) Router-branch signal statistics for all blocks.
  2) T-sensitivity test via fixed-routing temporal collapse.
  3) Counterfactual routing via expert-axis permutation/swap.

Outputs CSV reports under:
  analysis/temporal_merging/<exp_name>/
"""

import argparse
import ast
import csv
import os
import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from spikingjelly.activation_based import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_caltech101 import NCaltech101
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model

try:
    from timm.models import clean_state_dict
except Exception:
    from timm.models.helpers import clean_state_dict

import dvs_utils
import model  # noqa: F401, registers sdt


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default="", type=str)

    p = argparse.ArgumentParser()
    p.add_argument("-data-dir", default="/scratch1/bkrhee/data", type=str)
    p.add_argument("--dataset", "-d", default="gesture", type=str)
    p.add_argument("--val-split", default="validation", type=str)
    p.add_argument("--model", default="sdt", type=str)
    p.add_argument("--pooling_stat", default="1111", type=str)
    p.add_argument("--spike-mode", default="lif", type=str)
    p.add_argument("--layer", default=2, type=int)
    p.add_argument("--in-channels", default=2, type=int)
    p.add_argument("--num-classes", type=int, default=11)
    p.add_argument("--time-steps", type=int, default=10)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--num-experts", type=int, default=2)
    p.add_argument("--expert-timesteps", default=None)
    p.add_argument("--only-expert-ids", default=None)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--drop-block", type=float, default=None)
    p.add_argument("--crop-pct", type=float, default=None)
    p.add_argument("--mean", type=float, nargs="+", default=None)
    p.add_argument("--std", type=float, nargs="+", default=None)
    p.add_argument("--interpolation", default="", type=str)
    p.add_argument("--TET", default=False, type=bool)
    p.add_argument("--val-batch-size", "-vb", type=int, default=16)
    p.add_argument("--workers", "-j", type=int, default=4)
    p.add_argument("--resume", required=True, type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="./analysis/temporal_merging")
    p.add_argument("--experiment-name", type=str, default="temporal_merging_run")
    p.add_argument("--compute-isi", action="store_true", default=False)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            p.set_defaults(**cfg)
    args = p.parse_args(remaining)
    args.config = args_config.config
    return args


def normalize_only_expert_ids(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]
        return [int(parsed)]
    return [int(value)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(args):
    only_expert_ids = normalize_only_expert_ids(getattr(args, "only_expert_ids", None))
    dvs_mode = args.dataset in ["cifar10-dvs", "cifar10-dvs-tet", "gesture", "ncaltech101"]
    model_inst = create_model(
        args.model,
        T=args.time_steps,
        pretrained=False,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        pooling_stat=args.pooling_stat,
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        embed_dims=args.dim,
        mlp_ratios=args.mlp_ratio,
        num_experts=args.num_experts,
        expert_timesteps=getattr(args, "expert_timesteps", None),
        in_channels=args.in_channels,
        qkv_bias=False,
        depths=args.layer,
        sr_ratios=1,
        spike_mode=args.spike_mode,
        dvs_mode=dvs_mode,
        TET=args.TET,
    )
    for module in model_inst.modules():
        if hasattr(module, "only_expert_ids"):
            module.only_expert_ids = only_expert_ids

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = clean_state_dict(ckpt["state_dict"])
    else:
        state_dict = ckpt
    result = model_inst.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[WARN] Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARN] Unexpected keys: {result.unexpected_keys}")

    model_inst = model_inst.to(args.device)
    model_inst.eval()
    return model_inst


def make_loader(args):
    if args.dataset == "gesture":
        ds = DVS128Gesture(
            args.data_dir, train=False, data_type="frame",
            frames_number=args.time_steps, split_by="number",
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    if args.dataset == "cifar10-dvs":
        ds = CIFAR10DVS(
            args.data_dir, data_type="frame", frames_number=args.time_steps,
            split_by="number", transform=dvs_utils.Resize(64)
        )
        _, ds_eval = dvs_utils.split_to_train_test_set(0.9, ds, 10)
        return torch.utils.data.DataLoader(
            ds_eval, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    if args.dataset == "ncaltech101":
        _ = NCaltech101(
            args.data_dir, data_type="frame", frames_number=args.time_steps, split_by="number"
        )
        _, ds_eval = dvs_utils.build_ncaltech(args.data_dir, True)
        return torch.utils.data.DataLoader(
            ds_eval, batch_size=args.val_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        batch_size=args.val_batch_size
    )
    data_config = resolve_data_config(vars(args), model=None)
    loader = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=args.val_batch_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=True,
    )
    return loader


class RouterCapture:
    def __init__(self, net):
        self.gate_inputs = {}
        self.gate_spikes = {}
        self.routing_idx = {}
        self.routing_conf = {}
        self.handles = []
        self._register(net)

    def _register(self, net):
        import re
        gate_pat = re.compile(r"^block\.(\d+)\.mlp\.gate$")
        lif_pat = re.compile(r"^block\.(\d+)\.mlp\.gate\.gate_lif1$")
        for name, module in net.named_modules():
            gm = gate_pat.match(name)
            if gm:
                bid = int(gm.group(1))
                self.handles.append(module.register_forward_hook(self._gate_hook(bid)))
            lm = lif_pat.match(name)
            if lm:
                bid = int(lm.group(1))
                self.handles.append(module.register_forward_hook(self._lif_hook(bid)))

    def _gate_hook(self, block_id):
        def hook_fn(module, inputs, output):
            x_in = inputs[0].detach()
            self.gate_inputs[block_id] = x_in
            if getattr(module, "last_indices", None) and getattr(module, "last_raw_gates", None) is not None:
                idx = module.last_indices[0].detach()
                raw = module.last_raw_gates.detach()
                conf = raw.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                self.routing_idx[block_id] = idx
                self.routing_conf[block_id] = conf
        return hook_fn

    def _lif_hook(self, block_id):
        def hook_fn(module, inputs, output):
            self.gate_spikes[block_id] = output.detach()
        return hook_fn

    def clear_batch(self):
        self.gate_inputs.clear()
        self.gate_spikes.clear()
        self.routing_idx.clear()
        self.routing_conf.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def temporal_entropy_from_signal(sig_t):
    # sig_t: (..., T) non-negative
    eps = 1e-8
    p = sig_t / (sig_t.sum(dim=-1, keepdim=True) + eps)
    ent = -(p * (p + eps).log()).sum(dim=-1)
    norm = np.log(sig_t.shape[-1]) if sig_t.shape[-1] > 1 else 1.0
    return ent / norm


def mean_isi_and_cv(binary_spike_t):
    # binary_spike_t: (T,) {0,1}
    idx = torch.where(binary_spike_t > 0)[0]
    if idx.numel() < 2:
        return float("nan"), float("nan")
    diff = idx[1:] - idx[:-1]
    m = diff.float().mean().item()
    s = diff.float().std(unbiased=False).item()
    cv = s / (m + 1e-8)
    return m, cv


def get_expert_timesteps(model_inst):
    for m in model_inst.modules():
        if hasattr(m, "expert_timesteps"):
            return list(m.expert_timesteps)
    return None


def forward_logits_from_features(model_inst, feat_tbd):
    # feat_tbd: (T,B,D)
    functional.reset_net(model_inst.head_lif)
    x = model_inst.head_lif(feat_tbd)
    x = model_inst.head(x)
    return x.mean(0)


def run_exp1_router_stats(model_inst, loader, args, capture):
    rows = []
    with torch.no_grad():
        for bi, (images, _) in enumerate(loader):
            images = images.float().to(args.device)
            functional.reset_net(model_inst)
            capture.clear_batch()
            _ = model_inst(images, hook={})

            for block_id in sorted(capture.routing_idx.keys()):
                if block_id not in capture.gate_inputs or block_id not in capture.gate_spikes:
                    continue
                idx = capture.routing_idx[block_id]         # (B,N)
                conf = capture.routing_conf[block_id]       # (B,N)
                x_in = capture.gate_inputs[block_id]        # (T,B,D,H,W)
                spk = capture.gate_spikes[block_id]         # (T,B,D,H,W)
                T, B, D, H, W = x_in.shape
                N = H * W

                x_tok = x_in.flatten(3).permute(1, 3, 0, 2).contiguous()   # (B,N,T,D)
                s_tok = spk.flatten(3).permute(1, 3, 0, 2).contiguous()     # (B,N,T,D)

                vmem_abs = x_tok.abs().mean(dim=(2, 3))                     # (B,N)
                vmem_var_t = x_tok.mean(dim=-1).var(dim=-1, unbiased=False) # (B,N)
                vmem_t_energy = x_tok.abs().mean(dim=-1)                    # (B,N,T)
                vmem_ent = temporal_entropy_from_signal(vmem_t_energy)

                spike_rate = s_tok.mean(dim=(2, 3))                         # (B,N)
                spike_t = (s_tok.mean(dim=-1) > 0).float()                  # (B,N,T)
                spike_ent = temporal_entropy_from_signal(spike_t + 1e-6)

                unique_e = torch.unique(idx).tolist()
                for e in unique_e:
                    m = idx.eq(int(e))
                    if m.sum().item() == 0:
                        continue
                    row = {
                        "batch": bi,
                        "block": block_id,
                        "expert": int(e),
                        "token_count": int(m.sum().item()),
                        "router_conf_mean": float(conf[m].mean().item()),
                        "vmem_abs_mean": float(vmem_abs[m].mean().item()),
                        "vmem_var_t_mean": float(vmem_var_t[m].mean().item()),
                        "vmem_entropy_mean": float(vmem_ent[m].mean().item()),
                        "spike_rate_mean": float(spike_rate[m].mean().item()),
                        "spike_entropy_mean": float(spike_ent[m].mean().item()),
                    }
                    if args.compute_isi:
                        isi_vals, cv_vals = [], []
                        where = torch.where(m)
                        for k in range(where[0].numel()):
                            b_idx = where[0][k]
                            n_idx = where[1][k]
                            mi, cv = mean_isi_and_cv(spike_t[b_idx, n_idx])
                            if not np.isnan(mi):
                                isi_vals.append(mi)
                                cv_vals.append(cv)
                        row["isi_mean"] = float(np.mean(isi_vals)) if isi_vals else float("nan")
                        row["isi_cv_mean"] = float(np.mean(cv_vals)) if cv_vals else float("nan")
                    rows.append(row)

            if args.max_batches is not None and (bi + 1) >= args.max_batches:
                break
    return rows


@contextmanager
def temporary_hooks(hooks):
    handles = [h() for h in hooks]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def run_eval_accuracy(model_inst, loader, args):
    total, correct = 0, 0
    with torch.no_grad():
        for bi, (images, targets) in enumerate(loader):
            images = images.float().to(args.device)
            targets = targets.to(args.device)
            functional.reset_net(model_inst)
            logits, _ = model_inst(images, hook={})
            pred = logits.argmax(dim=-1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            if args.max_batches is not None and (bi + 1) >= args.max_batches:
                break
    return correct / max(total, 1), total


def build_t_collapse_hook(block_id, which="highT"):
    def registrar(model_inst):
        blk = model_inst.block[block_id].mlp

        def hook_fn(module, inputs, output):
            out, loss, hook = output
            idx = module.gate.last_indices[0] if module.gate.last_indices else None
            if idx is None:
                return output
            timesteps = module.expert_timesteps
            high_e = int(np.argmax(timesteps))
            low_e = int(np.argmin(timesteps))
            chosen_e = high_e if which == "highT" else low_e

            T, B, D, H, W = out.shape
            mask = idx.eq(chosen_e).view(B, H, W).unsqueeze(0).unsqueeze(2).to(out.device)
            out0 = out[0:1].expand(T, -1, -1, -1, -1)
            out_mod = torch.where(mask.bool(), out0, out)
            return out_mod, loss, hook

        return blk.register_forward_hook(hook_fn)
    return registrar


def build_counterfactual_gate_hook(block_id, mode="swap_high_low", seed=0):
    rng = np.random.RandomState(seed)

    def registrar(model_inst):
        gate = model_inst.block[block_id].mlp.gate

        def hook_fn(module, inputs, output):
            dispatch, combine, loss = output
            E = dispatch.shape[2]
            if mode == "swap_high_low":
                timesteps = model_inst.block[block_id].mlp.expert_timesteps
                high_e = int(np.argmax(timesteps))
                low_e = int(np.argmin(timesteps))
                perm = list(range(E))
                perm[high_e], perm[low_e] = perm[low_e], perm[high_e]
            else:
                perm = rng.permutation(E).tolist()
            dispatch2 = dispatch[:, :, perm, :]
            combine2 = combine[:, :, perm, :]
            return dispatch2, combine2, loss

        return gate.register_forward_hook(hook_fn)
    return registrar


def run_exp2_t_sensitivity(model_inst, loader, args):
    rows = []
    base_acc, n = run_eval_accuracy(model_inst, loader, args)
    for b in range(args.layer):
        for which in ["highT", "lowT"]:
            with temporary_hooks([build_t_collapse_hook(b, which=which)]):
                acc, _ = run_eval_accuracy(model_inst, loader, args)
            rows.append({
                "block": b,
                "condition": f"collapse_{which}_tokens_to_T1",
                "acc": acc,
                "acc_drop_vs_base": base_acc - acc,
                "num_samples": n,
            })
    return base_acc, rows


def run_exp3_counterfactual(model_inst, loader, args):
    rows = []
    base_acc, n = run_eval_accuracy(model_inst, loader, args)
    for b in range(args.layer):
        for mode in ["swap_high_low", "permute_experts_random"]:
            with temporary_hooks([build_counterfactual_gate_hook(b, mode=mode, seed=args.seed + b)]):
                acc, _ = run_eval_accuracy(model_inst, loader, args)
            rows.append({
                "block": b,
                "counterfactual_mode": mode,
                "acc": acc,
                "acc_drop_vs_base": base_acc - acc,
                "num_samples": n,
            })
    return base_acc, rows


def aggregate_router_stats(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["block"], r["expert"])
        for k, v in r.items():
            if k in ["batch", "block", "expert"]:
                continue
            grouped[key][k].append(v)
    out = []
    for (b, e), vals in sorted(grouped.items()):
        row = {"block": b, "expert": e}
        for k, arr in vals.items():
            arr_np = np.array(arr, dtype=np.float64)
            row[f"{k}_mean"] = float(np.nanmean(arr_np))
        out.append(row)
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_root = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(out_root, exist_ok=True)

    print("[1/6] Load model and data ...")
    model_inst = load_model(args)
    loader = make_loader(args)
    capture = RouterCapture(model_inst)

    print("[2/6] Exp1 router-branch statistics ...")
    exp1_raw = run_exp1_router_stats(model_inst, loader, args, capture)
    exp1_summary = aggregate_router_stats(exp1_raw)
    write_csv(os.path.join(out_root, "router_stats_raw.csv"), exp1_raw)
    write_csv(os.path.join(out_root, "router_stats_summary.csv"), exp1_summary)

    print("[3/6] Exp2 T-sensitivity (fixed routing, token collapse) ...")
    base_acc_2, exp2_rows = run_exp2_t_sensitivity(model_inst, loader, args)
    write_csv(os.path.join(out_root, "t_sensitivity.csv"), exp2_rows)

    print("[4/6] Exp3 counterfactual routing ...")
    base_acc_3, exp3_rows = run_exp3_counterfactual(model_inst, loader, args)
    write_csv(os.path.join(out_root, "counterfactual_routing.csv"), exp3_rows)

    print("[5/6] Summary ...")
    summary = [{
        "base_acc_exp2": base_acc_2,
        "base_acc_exp3": base_acc_3,
        "num_exp1_rows": len(exp1_raw),
        "num_exp2_rows": len(exp2_rows),
        "num_exp3_rows": len(exp3_rows),
        "layer": args.layer,
        "num_experts": args.num_experts,
        "time_steps": args.time_steps,
    }]
    write_csv(os.path.join(out_root, "summary_all_blocks.csv"), summary)

    capture.remove()
    print("[6/6] Done.")
    print(f"Outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
