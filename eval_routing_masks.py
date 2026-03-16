"""Evaluate impact of last-block MoE routing on classification accuracy.

We compare three cases on the CIFAR-100 validation set:

  1) Case A (full expert only):
     - Keep only tokens routed to the expert with T=4 timesteps in the last block.
     - All other spatial tokens at that block are zeroed before the head.

  2) Case B (non-full experts only):
     - Keep only tokens NOT routed to the T=4 expert.

  3) Case C (random tokens, same count as Case A):
     - For each sample, keep the same number of random tokens as in Case A, regardless
       of which expert they were routed to.

All three cases reuse the backbone features from a single forward pass per batch,
but for each case we:
  - mask the last-block output (T,B,C,H,W) in the spatial dimension
  - pass it through head_lif and head, with head_lif state reset between cases

Usage:
  python eval_routing_masks.py \\
    -c conf/cifar100/4_384_300E_t4.yml \\
    --resume /path/to/model_best.pth.tar \\
    --val-batch-size 128 \\
    --data-dir /dataset/CIFAR100
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
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
    p.add_argument("--mlp-ratio", type=int, default=4)
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


class LastBlockCapture:
    """Forward hooks to capture last block output and its gate routing."""

    def __init__(self, net, last_block_idx):
        self.last_block_idx = last_block_idx
        self.x_last = None  # (T,B,C,H,W)
        self.indices_last = None  # (B,N) expert id per token
        self.H = None
        self.W = None
        self._handles = []
        self._register(net)

    def _register(self, net):
        # Capture last block output (after MoE + residual)
        blk_name = f"block.{self.last_block_idx}"
        gate_name = f"block.{self.last_block_idx}.mlp.gate"

        for name, module in net.named_modules():
            if name == blk_name:
                self._handles.append(
                    module.register_forward_hook(self._make_block_hook())
                )
            if name == gate_name:
                self._handles.append(
                    module.register_forward_hook(self._make_gate_hook())
                )

    def _make_block_hook(self):
        def hook_fn(module, inputs, output):
            # output: (T,B,C,H,W) from MS_Block_Conv.forward
            x, attn, hook = output
            self.x_last = x.detach()
            T, B, C, H, W = x.shape
            self.H, self.W = H, W

        return hook_fn

    def _make_gate_hook(self):
        def hook_fn(module, inputs, output):
            # module is Top2Gating; last_indices cached there
            if getattr(module, "last_indices", None) is None or not module.last_indices:
                return
            idx = module.last_indices[0]  # (B,N)
            if idx.ndim != 2:
                return
            self.indices_last = idx.detach()

        return hook_fn

    def clear(self):
        self.x_last = None
        self.indices_last = None
        self.H = None
        self.W = None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def mask_and_head_forward_case(
    model, x_last, mask_hw_bool, case_name="caseA", eps=1e-6
):
    """Apply spatial mask (B,H,W) to last-block features and run through head.

    - model: SpikeDrivenTransformer
    - x_last: (T,B,C,H,W) last block output
    - mask_hw_bool: (B,H,W) bool, True for tokens to keep
    - returns: logits (B,num_classes)

    LIF state is reset before each call.
    """
    T, B, C, H, W = x_last.shape
    device = x_last.device

    # Broadcast mask to (T,B,C,H,W)
    m = mask_hw_bool.to(device=device, dtype=x_last.dtype)
    m = m.unsqueeze(0).unsqueeze(2)  # (1,B,1,H,W)
    x_masked = x_last * m  # (T,B,C,H,W)

    # If a sample has no True tokens, fall back to original x_last for stability
    num_kept = mask_hw_bool.view(B, -1).sum(dim=1)  # (B,)
    if (num_kept == 0).any():
        # Build per-sample mixing: where num_kept==0 -> use original, else use masked
        mix = (num_kept > 0).to(device=device, dtype=x_last.dtype)  # (B,)
        mix = mix.view(1, B, 1, 1, 1)
        x_masked = mix * x_masked + (1.0 - mix) * x_last

    # As in SpikeDrivenTransformer.forward_features / forward
    x_TBC = x_masked.flatten(3).mean(3)  # (T,B,C)

    # Reset only head_lif before each case
    functional.reset_net(model.head_lif)
    x_lif = model.head_lif(x_TBC)  # (T,B,C)
    logits = model.head(x_lif)  # (T,B,num_classes)
    # For non-TET: mean over T
    logits = logits.mean(0)  # (B,num_classes)
    return logits


def evaluate(args):
    # Seeding for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    net = load_model_and_checkpoint(args)
    loader = make_loader(args)

    last_block_idx = args.layer - 1
    capture = LastBlockCapture(net, last_block_idx=last_block_idx)

    total = 0
    correct_A = 0      # T=4 expert only
    correct_A_half = 0 # T=4 expert tokens, half of them
    correct_B = 0      # non T=4 experts only
    correct_C = 0      # random tokens with same count as A
    correct_C_half = 0 # random tokens with same count as A_half
    # For logging average token assignment ratios and overlaps
    sum_ratio_full = 0.0
    sum_ratio_not_full = 0.0
    num_batches = 0
    sum_overlap_A_C = 0.0  # mean over samples of |A ∩ C| / |A|

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.float().to(args.device)
            targets = targets.to(args.device)

            functional.reset_net(net)
            capture.clear()

            # Run full model once to populate capture.x_last and capture.indices_last
            _ = net(images, hook=dict())

            x_last = capture.x_last  # (T,B,C,H,W)
            indices_last = capture.indices_last  # (B,N)
            H, W = capture.H, capture.W

            if x_last is None or indices_last is None:
                print(
                    f"[WARN] Missing capture outputs at batch {batch_idx}; skipping batch."
                )
                continue

            T, B, C, H, W = x_last.shape
            N = H * W

            # Expert with full timesteps: choose argmax over expert_timesteps
            # (assuming all MoEs share same expert_timesteps; read from first instance)
            full_expert_id = 0
            for name, module in net.named_modules():
                if hasattr(module, "expert_timesteps"):
                    timesteps = list(module.expert_timesteps)
                    full_expert_id = int(np.argmax(timesteps))
                    break

            # indices_last: (B,N), expert id per token
            idx = indices_last  # (B,N)
            mask_full = idx.eq(full_expert_id)  # (B,N)
            mask_not_full = idx.ne(full_expert_id)  # (B,N)

            # Per-batch average token ratios
            tokens_full_per_sample = mask_full.sum(dim=1).float()  # (B,)
            tokens_not_full_per_sample = mask_not_full.sum(dim=1).float()  # (B,)
            ratio_full_per_sample = tokens_full_per_sample / float(N)
            ratio_not_full_per_sample = tokens_not_full_per_sample / float(N)
            batch_ratio_full = ratio_full_per_sample.mean().item()
            batch_ratio_not_full = ratio_not_full_per_sample.mean().item()

            sum_ratio_full += batch_ratio_full
            sum_ratio_not_full += batch_ratio_not_full
            num_batches += 1

            mask_full_hw = mask_full.view(B, H, W)
            mask_not_full_hw = mask_not_full.view(B, H, W)

            # Case A': T=4 expert tokens, but keep only half (per sample, random)
            mask_full_half = torch.zeros_like(mask_full)
            # Case C: random same-count tokens per sample (based on mask_full)
            mask_rand = torch.zeros_like(mask_full)
            # Case C': random tokens with same count as A'
            mask_rand_half = torch.zeros_like(mask_full)

            for b in range(B):
                k = int(mask_full[b].sum().item())
                if k <= 0:
                    continue

                # A (already defined by mask_full); A' half of A
                k_half = max(1, k // 2)
                full_indices = mask_full[b].nonzero(as_tuple=False).view(-1)
                perm_full = torch.randperm(full_indices.numel(), device=idx.device)
                chosen_full_half = full_indices[perm_full[:k_half]]
                mask_full_half[b, chosen_full_half] = True

                # C: random k tokens over all N
                perm_all = torch.randperm(N, device=idx.device)
                chosen_all = perm_all[:k]
                mask_rand[b, chosen_all] = True

                # C': random k_half tokens over all N
                chosen_all_half = perm_all[k : k + k_half] if k + k_half <= N else perm_all[:k_half]
                mask_rand_half[b, chosen_all_half] = True

            mask_full_half_hw = mask_full_half.view(B, H, W)
            mask_rand_hw = mask_rand.view(B, H, W)
            mask_rand_half_hw = mask_rand_half.view(B, H, W)

            # Overlap between Case A and Case C (A 기준): |A ∩ C| / |A|
            overlap_AC = (mask_full & mask_rand).sum(dim=1).float()  # (B,)
            k_per_sample = mask_full.sum(dim=1).float()
            # Avoid division by zero: ignore samples with k==0 in this batch's average
            valid_mask = k_per_sample > 0
            if valid_mask.any():
                overlap_ratio_per_sample = overlap_AC[valid_mask] / k_per_sample[
                    valid_mask
                ]
                batch_overlap_A_C = overlap_ratio_per_sample.mean().item()
            else:
                batch_overlap_A_C = 0.0
            sum_overlap_A_C += batch_overlap_A_C

            # Case A
            logits_A = mask_and_head_forward_case(
                net, x_last, mask_full_hw, case_name="A"
            )
            # Case B
            logits_B = mask_and_head_forward_case(
                net, x_last, mask_not_full_hw, case_name="B"
            )
            # Case C
            logits_C = mask_and_head_forward_case(
                net, x_last, mask_rand_hw, case_name="C"
            )
            # Case A'
            logits_A_half = mask_and_head_forward_case(
                net, x_last, mask_full_half_hw, case_name="A_half"
            )
            # Case C'
            logits_C_half = mask_and_head_forward_case(
                net, x_last, mask_rand_half_hw, case_name="C_half"
            )

            pred_A = logits_A.argmax(dim=-1)
            pred_A_half = logits_A_half.argmax(dim=-1)
            pred_B = logits_B.argmax(dim=-1)
            pred_C = logits_C.argmax(dim=-1)
            pred_C_half = logits_C_half.argmax(dim=-1)

            correct_A += pred_A.eq(targets).sum().item()
            correct_A_half += pred_A_half.eq(targets).sum().item()
            correct_B += pred_B.eq(targets).sum().item()
            correct_C += pred_C.eq(targets).sum().item()
            correct_C_half += pred_C_half.eq(targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[batch {batch_idx+1}] "
                    f"AccA={correct_A/total:.4f}, "
                    f"AccA'={correct_A_half/total:.4f}, "
                    f"AccB={correct_B/total:.4f}, "
                    f"AccC={correct_C/total:.4f}, "
                    f"AccC'={correct_C_half/total:.4f}, "
                    f"ratio_full={batch_ratio_full:.4f}, "
                    f"ratio_not_full={batch_ratio_not_full:.4f}, "
                    f"overlap_A_C={batch_overlap_A_C:.4f}"
                )

    capture.remove()

    accA = correct_A / max(total, 1)
    accA_half = correct_A_half / max(total, 1)
    accB = correct_B / max(total, 1)
    accC = correct_C / max(total, 1)
    accC_half = correct_C_half / max(total, 1)
    print("\n=== Routing-mask accuracy comparison (CIFAR-100 eval set) ===")
    print(f"Case A   (T=4 expert only)          : {accA:.4f}")
    print(f"Case A'  (T=4 expert, half tokens)  : {accA_half:.4f}")
    print(f"Case B   (non T=4 experts only)     : {accB:.4f}")
    print(f"Case C   (random, |C| = |A|)        : {accC:.4f}")
    print(f"Case C'  (random, |C'| = |A'|)      : {accC_half:.4f}")
    if num_batches > 0:
        mean_ratio_full = sum_ratio_full / num_batches
        mean_ratio_not_full = sum_ratio_not_full / num_batches
        mean_overlap_A_C = sum_overlap_A_C / num_batches
        print("\n=== Average token assignment ratios (last block) ===")
        print(f"Mean ratio T=4 expert tokens      : {mean_ratio_full:.4f}")
        print(f"Mean ratio non T=4 expert tokens  : {mean_ratio_not_full:.4f}")
        print(
            "Mean overlap (A ∩ C) / |A|          : "
            f"{mean_overlap_A_C:.4f}  (A = T=4 expert tokens, C = random tokens)"
        )


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.resume) or ".", exist_ok=True)
    evaluate(args)


if __name__ == "__main__":
    main()

