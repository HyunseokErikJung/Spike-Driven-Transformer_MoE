"""Analyze MoE expert assignment ratios per block (raw vs capacity-adjusted).

For each transformer block and each expert, this script computes:
  - Raw assignment ratio: fraction of tokens routed to that expert by Top2Gating
                          before capacity constraints.
  - Capacity-adjusted ratio: fraction of tokens that actually get dispatched to
                             that expert after capacity/pruning.

It then generates bar plots (one per expert) comparing raw vs capacity ratios
across blocks.

Usage example:
  python analyze_routing_stats.py \\
    -c conf/cifar100/4_384_300E_t4.yml \\
    --resume /path/to/model_best.pth.tar \\
    --data-dir /dataset/CIFAR100 \\
    --val-batch-size 128 \\
    --output-dir ./visual
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.models.helpers import clean_state_dict

import model  # registers 'sdt'


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default="", type=str)

    p = argparse.ArgumentParser()
    p.add_argument("-data-dir", default="/dataset/CIFAR100", type=str)
    p.add_argument("--dataset", "-d", default="torch/cifar100", type=str)
    p.add_argument("--val-split", default="validation", type=str)
    p.add_argument("--model", default="sdt", type=str)
    p.add_argument("--pooling_stat", default="1111", type=str)
    p.add_argument("--spike-mode", default="lif", type=str)
    p.add_argument("--layer", default=4, type=int)
    p.add_argument("--in-channels", default=3, type=int)
    p.add_argument("--num-classes", type=int, default=100, help="CIFAR-100")
    p.add_argument("--time-steps", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--expert-timesteps", default=None, help="from config: list of int. None = default.")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--drop-block", type=float, default=None)
    p.add_argument("--crop-pct", type=float, default=None)
    p.add_argument("--mean", type=float, nargs="+", default=None)
    p.add_argument("--std", type=float, nargs="+", default=None)
    p.add_argument("--interpolation", default="", type=str)
    p.add_argument("--TET", default=False, type=bool)
    p.add_argument("--batch-size", "-b", type=int, default=128)
    p.add_argument("--val-batch-size", "-vb", type=int, default=128)
    p.add_argument("--workers", "-j", type=int, default=8)
    p.add_argument("--no-prefetcher", action="store_true", default=False)

    p.add_argument("--resume", required=True, type=str, help="checkpoint path")
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="optional limit on number of batches for analysis (for speed)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./visual",
        help="directory to save bar plots",
    )

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        import yaml

        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            p.set_defaults(**cfg)
    args = p.parse_args(remaining)
    args.config = args_config.config
    return args


def load_model_and_checkpoint(args):
    m = create_model(
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
        dvs_mode=False,
        TET=args.TET,
    )

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = clean_state_dict(ckpt["state_dict"])
    else:
        state_dict = ckpt
    result = m.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"[WARN] Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARN] Unexpected keys: {result.unexpected_keys}")

    m = m.to(args.device)
    m.eval()
    return m


def make_loader(args):
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.val_batch_size,
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


class RoutingStatsCapture:
    """Capture raw and capacity-adjusted expert assignment per block."""

    def __init__(self, net):
        # block_id -> dict with accumulators
        self.raw_counts = defaultdict(lambda: None)  # (E,) per block
        self.cap_counts = defaultdict(lambda: None)  # (E,) per block
        self.total_tokens = defaultdict(float)  # scalar per block
        self.confidence_lists = defaultdict(list)  # block_id -> list of 1d arrays (top-1 prob per token)
        self.index_lists = defaultdict(list)      # block_id -> list of 1d arrays (expert id per token)
        self.num_experts = None
        self._handles = []
        self._register(net)

    def _register(self, net):
        import re

        gate_pat = re.compile(r"(^|.*\.)block\.(\d+)\.mlp\.gate$")
        for name, module in net.named_modules():
            m = gate_pat.match(name)
            if m is None:
                continue
            block_id = int(m.group(2))
            handle = module.register_forward_hook(self._make_gate_hook(block_id))
            self._handles.append(handle)

    def _make_gate_hook(self, block_id):
        def hook_fn(module, inputs, output):
            # module: Top2Gating
            # inputs[0]: x, shape (T,B,D,H,W)
            # output[0]: dispatch_tensor, shape (B, N, E, Ccap)
            if (
                getattr(module, "last_indices", None) is None
                or not module.last_indices
            ):
                return

            x_in = inputs[0]
            T, B, D, H, W = x_in.shape
            N = H * W

            dispatch_tensor, combine_tensor, _ = output  # (B,N,E,Ccap), ...
            # Raw assignment indices: list[top_k], we use top-1
            idx = module.last_indices[0]  # (B,N)
            if idx.ndim != 2:
                return

            num_experts = int(module.num_gates)
            if self.num_experts is None:
                self.num_experts = num_experts

            # Raw counts per expert: one-hot over idx, sum over (B,N)
            one_hot = F.one_hot(idx, num_experts).to(dispatch_tensor.device)  # (B,N,E)
            counts_raw = one_hot.sum(dim=(0, 1)).float()  # (E,)

            # Capacity-adjusted: token is considered assigned to expert e if any slot > 0
            dispatch_mask = dispatch_tensor.sum(dim=-1) > 0  # (B,N,E)
            counts_cap = dispatch_mask.sum(dim=(0, 1)).float()  # (E,)

            # Accumulate
            if self.raw_counts[block_id] is None:
                self.raw_counts[block_id] = counts_raw.detach().cpu()
                self.cap_counts[block_id] = counts_cap.detach().cpu()
            else:
                self.raw_counts[block_id] += counts_raw.detach().cpu()
                self.cap_counts[block_id] += counts_cap.detach().cpu()
            self.total_tokens[block_id] += float(B * N)

            # Router confidence: softmax prob of selected expert per token
            raw_gates = getattr(module, "last_raw_gates", None)
            if raw_gates is not None and raw_gates.shape == (B, N, num_experts):
                idx_long = idx.long().clamp(0, num_experts - 1)
                confidence = raw_gates.gather(
                    dim=-1, index=idx_long.unsqueeze(-1)
                ).squeeze(-1)
                self.confidence_lists[block_id].append(
                    confidence.detach().cpu().numpy().ravel()
                )
                self.index_lists[block_id].append(
                    idx_long.detach().cpu().numpy().ravel()
                )

        return hook_fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def analyze(args):
    # Seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    net = load_model_and_checkpoint(args)
    loader = make_loader(args)
    capture = RoutingStatsCapture(net)

    print("Collecting routing statistics over validation set ...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.float().to(args.device)
            functional.reset_net(net)
            _ = net(images, hook=dict())

            if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
                break

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx+1} batches")

    capture.remove()

    if not capture.raw_counts:
        print("No routing statistics collected.")
        return

    # Expert timesteps for legend (e.g. E0 (T=4))
    expert_timesteps = None
    for _, module in net.named_modules():
        if hasattr(module, "expert_timesteps"):
            expert_timesteps = list(module.expert_timesteps)
            break
    if expert_timesteps is None or len(expert_timesteps) < (capture.num_experts or 0):
        expert_timesteps = [None] * (capture.num_experts or 4)

    # Prepare arrays: blocks sorted by id
    block_ids = sorted(capture.raw_counts.keys())
    num_blocks = len(block_ids)
    num_experts = capture.num_experts or 0

    raw_ratios = np.zeros((num_blocks, num_experts), dtype=np.float32)
    cap_ratios = np.zeros((num_blocks, num_experts), dtype=np.float32)

    for i, bid in enumerate(block_ids):
        total_tok = capture.total_tokens[bid]
        if total_tok <= 0:
            continue
        counts_raw = capture.raw_counts[bid].numpy()  # (E,)
        counts_cap = capture.cap_counts[bid].numpy()  # (E,)
        raw_ratios[i] = counts_raw / float(total_tok)
        cap_ratios[i] = counts_cap / float(total_tok)

    os.makedirs(args.output_dir, exist_ok=True)

    # Single plot: per block, two stacked bars side-by-side (raw, cap)
    import matplotlib.pyplot as plt

    expert_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    width = 0.38
    # x positions: block i -> raw at 2*i, cap at 2*i+1
    x_raw = np.array([2 * i for i in range(num_blocks)], dtype=float)
    x_cap = np.array([2 * i + 1 for i in range(num_blocks)], dtype=float)
    tick_positions = (x_raw + x_cap) / 2
    tick_labels = [str(bid) for bid in block_ids]

    fig, ax = plt.subplots(figsize=(max(6, num_blocks * 2), 4))

    bottom_raw = np.zeros(num_blocks)
    bottom_cap = np.zeros(num_blocks)
    for e in range(num_experts):
        color = expert_colors[e % len(expert_colors)]
        ts = expert_timesteps[e] if e < len(expert_timesteps) and expert_timesteps[e] is not None else "?"
        label = f"E{e} (T={ts})"
        ax.bar(x_raw, raw_ratios[:, e], width, bottom=bottom_raw, label=label, color=color)
        ax.bar(x_cap, cap_ratios[:, e], width, bottom=bottom_cap, color=color)
        bottom_raw += raw_ratios[:, e]
        bottom_cap += cap_ratios[:, e]

    ax.set_xlabel("Block index")
    ax.set_ylabel("Token ratio")
    ax.set_title("Expert token assignment (per block: left = raw, right = cap)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc="upper right", ncol=min(num_experts, 4))
    ax.set_ylim(0.0, 1.05)

    out_path = os.path.join(args.output_dir, "routing_stacked.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Router confidence: mean and histogram per block
    if capture.confidence_lists:
        block_ids_conf = sorted(capture.confidence_lists.keys())
        n_conf_blocks = len(block_ids_conf)
        if n_conf_blocks > 0:
            n_cols = min(n_conf_blocks, 4)
            n_rows = (n_conf_blocks + n_cols - 1) // n_cols
            fig2, axes = plt.subplots(
                n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
            )
            axes = axes.flatten()
            for ii, bid in enumerate(block_ids_conf):
                arr = np.concatenate(capture.confidence_lists[bid], axis=0)
                idx_flat = np.concatenate(capture.index_lists[bid], axis=0)
                arr = np.clip(arr, 1e-6, 1.0)
                mean_conf = float(np.mean(arr))
                ax = axes[ii]
                ax.hist(arr, bins=50, range=(0.0, 1.0), color="#4C72B0", alpha=0.8, edgecolor="none")
                ax.axvline(mean_conf, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_conf:.3f}")
                ax.set_xlabel("Top-1 probability")
                ax.set_ylabel("Count")
                ax.set_title(f"Block {bid} (mean={mean_conf:.3f})")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_xlim(0.0, 1.0)
                # Per-expert mean confidence (when that expert was selected) at bottom
                num_e = capture.num_experts or idx_flat.max() + 1
                lines = []
                for e in range(num_e):
                    mask = idx_flat == e
                    if mask.any():
                        mean_e = float(np.mean(arr[mask]))
                        lines.append(f"E{e}: {mean_e:.3f}")
                    else:
                        lines.append(f"E{e}: —")
                ax.text(
                    0.5, 0.02, "\n".join(lines),
                    transform=ax.transAxes, fontsize=8, va="bottom", ha="center",
                    family="monospace",
                )
            for jj in range(ii + 1, len(axes)):
                axes[jj].set_visible(False)
            out_conf = os.path.join(args.output_dir, "routing_confidence.png")
            fig2.tight_layout()
            fig2.savefig(out_conf, dpi=200)
            plt.close(fig2)
            print(f"Saved: {out_conf}")

    # Optional: print a small summary table
    print("\n=== Routing ratios summary (per block, per expert) ===")
    for i, bid in enumerate(block_ids):
        print(f"Block {bid}:")
        for e in range(num_experts):
            print(
                f"  Expert {e}: raw={raw_ratios[i, e]:.4f}, cap={cap_ratios[i, e]:.4f}"
            )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    analyze(args)


if __name__ == "__main__":
    main()

