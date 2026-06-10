import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloaders.trajectory_loader import get_push_train_val


def _flatten_valid(data: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return data[masks.bool()]


def _usage(indices: torch.Tensor, codebook_size: int) -> None:
    for q in range(indices.shape[-1]):
        counts = torch.bincount(
            indices[..., q].reshape(-1).cpu(), minlength=codebook_size
        ).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * torch.log(nz)).sum())
        print(
            f"q{q}: hist={counts.long().tolist()} "
            f"active={(counts > 0).sum().item():.0f}/{codebook_size} "
            f"perplexity={perplexity.item():.2f} "
            f"max_usage_ratio={probs.max().item():.4f}"
        )


def _force_module_device(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    module = module.to(device)
    for submodule in module.modules():
        if hasattr(submodule, "device"):
            submodule.device = device
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--data-directory",
        default="/home/nullptr/projects/bet/datasets/extracted/bet_data_release/blockpush",
    )
    parser.add_argument("--max-seqs", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    args = parser.parse_args()

    device = torch.device(args.device)
    snapshot = torch.load(args.snapshot, map_location=device)
    action_ae = _force_module_device(snapshot["action_ae"], device).eval()
    obs_encoding_net = _force_module_device(snapshot["obs_encoding_net"], device).eval()
    state_prior = _force_module_device(snapshot["state_prior"], device).eval()

    data_dir = Path(args.data_directory)
    observations = torch.from_numpy(
        np.load(data_dir / "multimodal_push_observations.npy")
    ).float()
    actions = torch.from_numpy(np.load(data_dir / "multimodal_push_actions.npy")).float()
    masks = torch.from_numpy(np.load(data_dir / "multimodal_push_masks.npy")).float()

    observations = observations[: args.max_seqs].to(device)
    actions = actions[: args.max_seqs].to(device)
    masks = masks[: args.max_seqs].to(device)

    with torch.no_grad():
        flat_actions = _flatten_valid(actions, masks)
        print("== action stats ==")
        print("mean", flat_actions.mean(dim=0).cpu().tolist())
        print("std", flat_actions.std(dim=0).cpu().tolist())
        print("min", flat_actions.min(dim=0).values.cpu().tolist())
        print("max", flat_actions.max(dim=0).values.cpu().tolist())

        latents = action_ae.encode_into_latent(actions)
        codes, offset_targets = latents if type(latents) == tuple else (latents, None)
        codebook_size = action_ae.num_latents
        flat_codes = _flatten_valid(codes, masks)
        print("== tokenizer code usage ==")
        _usage(flat_codes.long(), codebook_size)

        centers = action_ae.decode_actions(codes)
        center_error = (actions - centers).abs()
        valid_center_error = _flatten_valid(center_error, masks)
        print("== tokenizer reconstruction ==")
        print("center_l1", valid_center_error.mean().item())
        print("center_mse", _flatten_valid((actions - centers).pow(2), masks).mean().item())
        if offset_targets is not None:
            print("offset_target_abs_mean", _flatten_valid(offset_targets.abs(), masks).mean().item())
            final = action_ae.decode_actions((codes, offset_targets))
            print("final_l1_with_gt_offset", _flatten_valid((actions - final).abs(), masks).mean().item())

        train_set, val_set = get_push_train_val(
            data_directory=args.data_directory,
            train_fraction=0.95,
            random_seed=42,
            device="cpu",
            window_size=args.window_size,
        )
        dataset = train_set if args.split == "train" else val_set
        window_obs = []
        window_actions = []
        window_masks = []
        for idx in range(min(args.max_windows, len(dataset))):
            obs, act, mask = dataset[idx]
            window_obs.append(obs)
            window_actions.append(act)
            window_masks.append(mask)
        observations = torch.stack(window_obs, dim=0).to(device)
        actions = torch.stack(window_actions, dim=0).to(device)
        masks = torch.stack(window_masks, dim=0).to(device)
        latents = action_ae.encode_into_latent(actions)
        codes, offset_targets = latents if type(latents) == tuple else (latents, None)

        obs_rep = obs_encoding_net(observations)
        output, loss, components = state_prior.get_latent_and_loss(
            obs_rep=obs_rep,
            target_latents=latents,
            return_loss_components=True,
        )
        logits, pred_offsets = output if type(output) == tuple else (output, None)
        pred_codes = logits.argmax(dim=-1).permute(1, 0, 2)
        valid_pred_codes = _flatten_valid(pred_codes, masks)

        print("== policy predicted code usage ==")
        _usage(valid_pred_codes.long(), codebook_size)

        print("== policy code accuracy ==")
        code_correct_masks = []
        for q in range(codes.shape[-1]):
            correct = pred_codes[..., q] == codes[..., q]
            code_correct_masks.append(correct)
            print(f"acc_code_{q + 1}", _flatten_valid(correct.float(), masks).mean().item())
            topk = min(3, logits.shape[-1])
            topk_codes = logits[:, :, q, :].topk(topk, dim=-1).indices.permute(1, 0, 2)
            topk_correct = (topk_codes == codes[..., q, None]).any(dim=-1)
            print(
                f"acc3_code_{q + 1}",
                _flatten_valid(topk_correct.float(), masks).mean().item(),
            )
        if len(code_correct_masks) >= 2:
            joint_correct = torch.stack(code_correct_masks, dim=-1).all(dim=-1)
            print(
                "acc_code_joint",
                _flatten_valid(joint_correct.float(), masks).mean().item(),
            )

        print("== policy losses / offset diagnostics ==")
        for key, value in components.items():
            print(key, float(value.detach().cpu()))
        print("total_loss", float(loss.detach().cpu()))
        if pred_offsets is not None:
            if pred_offsets.ndim == 4:
                pred_offsets_bt = pred_offsets.permute(1, 0, 2, 3)
            else:
                pred_offsets_bt = pred_offsets.permute(1, 0, 2)
            print(
                "offset_pred_abs_mean_valid",
                _flatten_valid(pred_offsets_bt.abs(), masks).mean().item(),
            )


if __name__ == "__main__":
    main()
