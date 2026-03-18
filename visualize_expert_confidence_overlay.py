"""Visualize top-1 router confidence (softmax value) as a heatmap overlay per layer.

Based on visualize_expert_assignment.py: same grid layout (rows=images, col0=original,
col1..=original + layer-wise overlay). Instead of expert index, each token is shown
by the softmax probability of the selected (top-1) expert, as a scalar heatmap
overlaid on the image (e.g. viridis: low=purple, high=yellow).

Example:
  python visualize_expert_confidence_overlay.py \\
    -c conf/cifar100/4_384_300E_t4.yml \\
    --resume /path/to/model_best.pth.tar \\
    --num-images 4 \\
    --output-dir ./visual
"""

import argparse
import os
import re
from collections import OrderedDict

import yaml
import numpy as np
import torch

from spikingjelly.clock_driven import functional
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.models.helpers import clean_state_dict

import model  # registers 'sdt'


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default="", type=str)

    p = argparse.ArgumentParser()
    p.add_argument("-data-dir", default="/scratch1/bkrhee/data", type=str)
    p.add_argument("--dataset", "-d", default="torch/cifar100", type=str)
    p.add_argument("--val-split", default="validation", type=str)
    p.add_argument("--model", default="sdt", type=str)
    p.add_argument("--pooling_stat", default="1111", type=str)
    p.add_argument("--spike-mode", default="lif", type=str)
    p.add_argument("--layer", default=4, type=int)
    p.add_argument("--in-channels", default=3, type=int)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--time-steps", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--expert-timesteps", default=None, help="from config: list of int. None = default.")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.2)
    p.add_argument("--drop-block", type=float, default=None)
    p.add_argument("--crop-pct", type=float, default=None)
    p.add_argument("--mean", type=float, nargs="+", default=None)
    p.add_argument("--std", type=float, nargs="+", default=None)
    p.add_argument("--interpolation", default="", type=str)
    p.add_argument("--TET", default=False, type=bool)
    p.add_argument("--batch-size", "-b", type=int, default=1)
    p.add_argument("--workers", "-j", type=int, default=2)

    p.add_argument("--resume", required=True, type=str, help="checkpoint path")
    p.add_argument("--num-images", type=int, default=4, help="number of images")
    p.add_argument("--start-idx", type=int, default=0, help="start index in val loader")
    p.add_argument("--output-dir", default="./expert_assign_vis", type=str)
    p.add_argument("--device", default="cuda:0", type=str)
    p.add_argument("--dpi", type=int, default=200, help="output DPI")
    p.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.6,
        help="alpha for confidence heatmap overlay (default: 0.6)",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="matplotlib colormap for confidence (default: viridis)",
    )

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
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
    ckpt = torch.load(args.resume, map_location="cpu")
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
        batch_size=1,
    )
    data_config = resolve_data_config(vars(args), model=None)
    loader = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=1,
        is_training=False,
        use_prefetcher=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
    )
    return loader


def get_class_names(args):
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=1,
    )
    if hasattr(dataset_eval, "classes"):
        return list(dataset_eval.classes)
    if hasattr(dataset_eval, "classnames"):
        return list(dataset_eval.classnames)
    if hasattr(dataset_eval, "class_to_idx"):
        idx_to_class = {v: k for k, v in dataset_eval.class_to_idx.items()}
        return [idx_to_class[i] for i in range(len(idx_to_class))]
    return None


class ConfidenceCapture:
    """Capture per-block top-1 softmax value (confidence) per token. Stores (H, W, conf_hw) float."""

    def __init__(self, net):
        self.assignments = OrderedDict()  # block_id -> (H, W, conf_hw)
        self._handles = []
        self._register(net)

    def _register(self, net):
        gate_pat = re.compile(r"(^|.*\.)block\.(\d+)\.mlp\.gate$")
        for name, module in net.named_modules():
            m = gate_pat.match(name)
            if m is None:
                continue
            block_id = int(m.group(2))
            self._handles.append(
                module.register_forward_hook(self._make_gate_hook(block_id))
            )

    def _make_gate_hook(self, block_id: int):
        def hook_fn(module, input, output):
            raw_gates = getattr(module, "last_raw_gates", None)
            indices = getattr(module, "last_indices", None)
            if raw_gates is None or not indices:
                return
            x_in = input[0]
            H, W = int(x_in.shape[-2]), int(x_in.shape[-1])
            B, N, E = raw_gates.shape
            if N != H * W:
                return
            idx = indices[0]  # (B, N)
            idx_long = idx.long().clamp(0, E - 1)
            conf = raw_gates.gather(dim=-1, index=idx_long.unsqueeze(-1)).squeeze(-1)
            conf0 = conf[0].detach().cpu().float().numpy()
            conf_hw = np.clip(conf0.reshape(H, W), 0.0, 1.0)
            self.assignments[block_id] = (H, W, conf_hw)

        return hook_fn

    def clear(self):
        self.assignments.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _denormalize_image(img_tensor, mean, std):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.cpu().float()
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).mul(255).byte()
    return img.permute(1, 2, 0).numpy()


def _resize_confidence_to_image(conf_hw, target_h, target_w):
    h, w = conf_hw.shape
    if h == target_h and w == target_w:
        return conf_hw
    if target_h % h != 0 or target_w % w != 0:
        sh, sw = max(1, target_h // h), max(1, target_w // w)
        out = np.repeat(np.repeat(conf_hw, sh, axis=0), sw, axis=1)
        if out.shape[0] > target_h or out.shape[1] > target_w:
            out = out[:target_h, :target_w]
        elif out.shape[0] < target_h or out.shape[1] < target_w:
            out = np.pad(
                out,
                ((0, target_h - out.shape[0]), (0, target_w - out.shape[1])),
                mode="edge",
            )
        return out
    sh, sw = target_h // h, target_w // w
    return np.repeat(np.repeat(conf_hw, sh, axis=0), sw, axis=1)


def _confidence_to_rgba(conf_map, cmap_name="viridis", alpha=0.6):
    """(H,W) float in [0,1] -> (H,W,4) RGBA for overlay."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(conf_map)
    if rgba.shape[-1] == 4:
        rgba = rgba.copy()
        rgba[:, :, 3] *= alpha
    else:
        rgba = np.concatenate([rgba, np.full((*conf_map.shape, 1), alpha)], axis=-1)
    return rgba


def save_grid_png(
    grid_maps,
    output_path: str,
    dpi: int = 200,
    original_images=None,
    overlay_alpha: float = 0.6,
    cmap_name: str = "viridis",
    row_pred_labels=None,
    row_gt_labels=None,
    row_confidences=None,
):
    """grid_maps: list[list[np.ndarray|None]] float (H,W) in [0,1] per layer."""
    import matplotlib.pyplot as plt

    R = len(grid_maps)
    num_layers = max(len(r) for r in grid_maps) if R > 0 else 0
    has_originals = (
        original_images is not None
        and len(original_images) == R
        and all(x is not None for x in original_images)
    )
    C = (1 if has_originals else 0) + num_layers
    if R == 0 or C == 0:
        raise RuntimeError("No maps to plot.")

    fig_w = max(6.0, 2.0 * C)
    fig_h = max(4.0, 1.6 * R)
    fig, axes = plt.subplots(R, C, figsize=(fig_w, fig_h), squeeze=False)

    for i in range(R):
        row_maps = grid_maps[i]
        img_rgb = original_images[i] if has_originals else None
        H_img = img_rgb.shape[0] if img_rgb is not None else None
        W_img = img_rgb.shape[1] if img_rgb is not None else None

        if has_originals:
            ax = axes[i, 0]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img_rgb)
            if i == 0:
                ax.set_title("original", fontsize=10)
            ax.set_ylabel(f"img {i}", fontsize=10)

        for j in range(num_layers):
            col = (1 if has_originals else 0) + j
            ax = axes[i, col]
            ax.set_xticks([])
            ax.set_yticks([])
            m = row_maps[j] if j < len(row_maps) else None
            if has_originals and img_rgb is not None:
                ax.imshow(img_rgb)
                if m is not None and H_img is not None and W_img is not None:
                    conf_big = _resize_confidence_to_image(m, H_img, W_img)
                    overlay = _confidence_to_rgba(
                        conf_big, cmap_name=cmap_name, alpha=overlay_alpha
                    )
                    ax.imshow(overlay, interpolation="bilinear")
            else:
                if m is None:
                    ax.set_facecolor("#f2f2f2")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
                else:
                    ax.imshow(m, cmap=cmap_name, vmin=0.0, vmax=1.0, interpolation="nearest")
            if i == 0:
                ax.set_title(f"layer {j}", fontsize=10)
            if not has_originals and col == 0:
                ax.set_ylabel(f"img {i}", fontsize=10)

        if (
            row_pred_labels is not None
            and row_gt_labels is not None
            and row_confidences is not None
        ):
            if i < len(row_pred_labels) and i < len(row_gt_labels) and i < len(row_confidences):
                ax_last = axes[i, C - 1]
                ax_last.text(1.02, 0.55, f"gt={row_gt_labels[i]}", transform=ax_last.transAxes, va="center", ha="left", fontsize=9, color="black")
                ax_last.text(1.02, 0.35, f"pred={row_pred_labels[i]} ({row_confidences[i]:.2f})", transform=ax_last.transAxes, va="center", ha="left", fontsize=9, color="black")

    # Reserve right margin for colorbar and row text; avoid tight_layout (incompatible with colorbar axes).
    fig.subplots_adjust(left=0.05, right=0.70, top=0.95, bottom=0.05, wspace=0.05, hspace=0.12)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="top-1 softmax")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.resume} ...")
    net = load_model_and_checkpoint(args)
    loader = make_loader(args)
    class_names = get_class_names(args)
    data_config = resolve_data_config(vars(args), model=None)
    img_mean = list(data_config["mean"])
    img_std = list(data_config["std"])

    capture = ConfidenceCapture(net)
    if not capture._handles:
        print("[ERROR] No gate modules found (block.X.mlp.gate).")
        return

    num_layers = int(args.layer)
    num_images = int(args.num_images)
    start_idx = int(args.start_idx)

    grid = []
    original_images = []
    row_pred_labels = []
    row_gt_labels = []
    row_confidences = []

    print(f"Running {num_images} image(s) starting at idx={start_idx} (top-1 confidence heatmap) ...")
    with torch.no_grad():
        seen = 0
        for i, (img, target) in enumerate(loader):
            if i < start_idx:
                continue
            if seen >= num_images:
                break

            img = img.float().to(args.device)
            orig_np = _denormalize_image(img, img_mean, img_std)
            original_images.append(orig_np)

            functional.reset_net(net)
            capture.clear()

            output, _ = net(img, hook=dict())
            pred = output.argmax(dim=-1).item()
            gt = target.item()
            probs = output.softmax(dim=-1)
            conf = probs.max(dim=-1).values.item()
            if class_names is not None and pred < len(class_names) and gt < len(class_names):
                pred_label = class_names[pred]
                gt_label = class_names[gt]
            else:
                pred_label = str(pred)
                gt_label = str(gt)
            row_pred_labels.append(pred_label)
            row_gt_labels.append(gt_label)
            row_confidences.append(conf)

            row = []
            for layer_id in range(num_layers):
                if layer_id in capture.assignments:
                    _, _, conf_hw = capture.assignments[layer_id]
                    row.append(conf_hw)
                else:
                    row.append(None)
            grid.append(row)
            seen += 1

    capture.remove_hooks()

    out_path = os.path.join(args.output_dir, "expert_confidence_heatmap.png")
    save_grid_png(
        grid,
        output_path=out_path,
        dpi=args.dpi,
        original_images=original_images,
        overlay_alpha=args.overlay_alpha,
        cmap_name=args.cmap,
        row_pred_labels=row_pred_labels,
        row_gt_labels=row_gt_labels,
        row_confidences=row_confidences,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
