from itertools import product
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from models.action_ae.discretizers.base import AbstractDiscretizer


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
) -> nn.Sequential:
    if hidden_depth <= 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualVQVAEActionDiscretizer(AbstractDiscretizer):
    """
    Residual VQ-VAE action tokenizer for single-step BeT actions.

    It mirrors KMeansDiscretizer's public interface while encoding each action as
    N_q independent code indices. N_q=1 is vanilla VQ; N_q=2 is residual VQ.
    """

    def __init__(
        self,
        action_dim: int,
        num_quantizers: int = 2,
        codebook_size: int = 16,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        hidden_depth: int = 1,
        device: Union[str, torch.device] = "cpu",
        predict_offsets: bool = True,
        train_steps: int = 1000,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        commitment_loss_scale: float = 1.0,
        codebook_loss_scale: float = 1.0,
        reconstruction_loss: str = "mse",
        normalize_actions: bool = True,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        dead_code_restart_interval: int = 50,
        dead_code_threshold: int = 0,
        usage_log_interval: int = 50,
        restart_noise_scale: float = 0.01,
        verbose: bool = True,
    ):
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        if codebook_size < 1:
            raise ValueError("codebook_size must be >= 1")
        if reconstruction_loss not in {"mse", "l1"}:
            raise ValueError("reconstruction_loss must be 'mse' or 'l1'")

        self.action_dim = action_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        self.device = torch.device(device)
        self.predict_offsets = predict_offsets
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.commitment_loss_scale = commitment_loss_scale
        self.codebook_loss_scale = codebook_loss_scale
        self.reconstruction_loss = reconstruction_loss
        self.normalize_actions = normalize_actions
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.dead_code_restart_interval = dead_code_restart_interval
        self.dead_code_threshold = dead_code_threshold
        self.usage_log_interval = usage_log_interval
        self.restart_noise_scale = restart_noise_scale
        self.verbose = verbose

        self.encoder = _build_mlp(action_dim, embedding_dim, hidden_dim, hidden_depth)
        self.decoder = _build_mlp(embedding_dim, action_dim, hidden_dim, hidden_depth)
        codebooks = torch.randn(num_quantizers, codebook_size, embedding_dim) * 0.02
        self.codebooks = nn.Parameter(codebooks)
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.training_losses = []
        self._maybe_init_identity_projection()
        self.to(self.device)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return module

    def fit_discretizer(self, input_actions: torch.Tensor) -> None:
        assert (
            input_actions.shape[-1] == self.action_dim
        ), f"Input action dimension {input_actions.shape[-1]} does not match configured action_dim {self.action_dim}"

        flattened_actions = input_actions.reshape(-1, self.action_dim).to(self.device)
        if flattened_actions.numel() == 0:
            raise ValueError("Cannot fit RVQ discretizer on an empty action tensor")

        self._fit_action_normalizer(flattened_actions)
        self._log_action_stats(flattened_actions)
        normalized_actions = self._normalize_actions(flattened_actions)
        if self.kmeans_init:
            self._initialize_codebooks_with_residual_kmeans(normalized_actions)
            self._print_diagnostics("after_kmeans_init", normalized_actions)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.train()
        self.training_losses = []
        iterator = tqdm.trange(self.train_steps, disable=not self.verbose)
        iterator.set_description("RVQ-VAE tokenizer")
        num_actions = normalized_actions.shape[0]
        for step in iterator:
            batch_indices = torch.randint(
                num_actions,
                (min(self.batch_size, num_actions),),
                device=self.device,
            )
            action_batch = normalized_actions[batch_indices]

            z_e = self.encoder(action_batch)
            z_q, indices, vq_loss, residual_inputs = self._quantize(
                z_e, return_residuals=True
            )
            reconstructed = self.decoder(z_q)
            recon_loss = self._reconstruction_loss(action_batch, reconstructed)
            loss = recon_loss + vq_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (
                self.dead_code_restart_interval > 0
                and (step + 1) % self.dead_code_restart_interval == 0
            ):
                self._restart_dead_codes(indices, residual_inputs)

            self.training_losses.append(
                {
                    "total": float(loss.detach().cpu()),
                    "reconstruction": float(recon_loss.detach().cpu()),
                    "vq": float(vq_loss.detach().cpu()),
                }
            )
            iterator.set_postfix_str(
                f"recon:{recon_loss.detach().item():.2e},vq:{vq_loss.detach().item():.2e}"
            )
            if (
                self.usage_log_interval > 0
                and (step + 1) % self.usage_log_interval == 0
            ):
                self._print_batch_diagnostics(
                    f"step_{step + 1}",
                    action_batch,
                    reconstructed.detach(),
                    indices.detach(),
                )
        self.eval()
        self._print_diagnostics("after_train", normalized_actions)

    @property
    def suggested_actions(self) -> torch.Tensor:
        if self.num_quantizers > 4:
            raise ValueError(
                "suggested_actions enumerates K ** N_q combinations; "
                "use decode_actions on explicit codes for large N_q."
            )
        code_values = [range(self.codebook_size) for _ in range(self.num_quantizers)]
        codes = torch.tensor(
            list(product(*code_values)),
            dtype=torch.long,
            device=self.device,
        )
        return self.decode_actions(codes).detach()

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert (
            input_action.shape[-1] == self.action_dim
        ), "Input action dimension does not match fitted model"

        original_shape = input_action.shape[:-1]
        flat_actions = input_action.reshape(-1, self.action_dim).to(self.device)
        normalized_actions = self._normalize_actions(flat_actions)
        with torch.no_grad():
            z_e = self.encoder(normalized_actions)
            _, flat_codes, _ = self._quantize(z_e)
            codes = flat_codes.reshape(original_shape + (self.num_quantizers,))

            if self.predict_offsets:
                reconstructed_action = self.decode_actions(codes)
                offsets = input_action.to(self.device) - reconstructed_action
                return (codes, offsets)
            return codes

    def decode_actions(
        self,
        latent_action_batch: torch.Tensor,
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        offsets = None
        if type(latent_action_batch) == tuple:
            latent_action_batch, offsets = latent_action_batch

        codes = latent_action_batch.to(self.device).long()
        if codes.shape[-1] != self.num_quantizers:
            if self.num_quantizers == 1:
                codes = codes.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Expected last latent dimension {self.num_quantizers}, got {codes.shape[-1]}"
                )

        original_shape = codes.shape[:-1]
        flat_codes = codes.reshape(-1, self.num_quantizers)
        z_q = self._codes_to_embeddings(flat_codes)
        reconstructed_action = self._denormalize_actions(self.decoder(z_q)).reshape(
            original_shape + (self.action_dim,)
        )
        if offsets is not None:
            reconstructed_action = reconstructed_action + offsets.to(self.device)
        return reconstructed_action

    def get_codebook_embeddings(self, detach: bool = True) -> torch.Tensor:
        codebooks = self.codebooks.detach() if detach else self.codebooks
        return codebooks

    def get_code_embedding_for_layer(
        self,
        layer_idx: int,
        code_ids: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        if layer_idx < 0 or layer_idx >= self.num_quantizers:
            raise ValueError(
                f"layer_idx must be in [0, {self.num_quantizers}), got {layer_idx}"
            )
        codebook = self.get_codebook_embeddings(detach=detach)[layer_idx]
        return F.embedding(code_ids.to(self.device).long(), codebook)

    def get_code_embeddings(
        self, codes: torch.Tensor, detach: bool = True
    ) -> torch.Tensor:
        codes = codes.to(self.device).long()
        if codes.shape[-1] != self.num_quantizers:
            if self.num_quantizers == 1:
                codes = codes.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Expected last code dimension {self.num_quantizers}, got {codes.shape[-1]}"
                )
        embeddings = [
            self.get_code_embedding_for_layer(
                quantizer_index, codes[..., quantizer_index], detach=detach
            )
            for quantizer_index in range(self.num_quantizers)
        ]
        return torch.stack(embeddings, dim=-2)

    def sample_latents(self, num_latents: Optional[int] = None) -> torch.Tensor:
        if num_latents is None:
            code_values = [range(self.codebook_size) for _ in range(self.num_quantizers)]
            return torch.tensor(
                list(product(*code_values)),
                dtype=torch.long,
                device=self.device,
            )
        codes = torch.randint(
            self.codebook_size,
            (num_latents, 1, self.num_quantizers),
            device=self.device,
        )
        return codes

    def _maybe_init_identity_projection(self) -> None:
        if self.hidden_depth != 0 or self.embedding_dim != self.action_dim:
            return
        encoder_layer = self.encoder[0]
        decoder_layer = self.decoder[0]
        if not isinstance(encoder_layer, nn.Linear) or not isinstance(
            decoder_layer, nn.Linear
        ):
            return
        with torch.no_grad():
            encoder_layer.weight.copy_(torch.eye(self.action_dim))
            encoder_layer.bias.zero_()
            decoder_layer.weight.copy_(torch.eye(self.action_dim))
            decoder_layer.bias.zero_()

    def _fit_action_normalizer(self, actions: torch.Tensor) -> None:
        if not self.normalize_actions:
            self.action_mean.zero_()
            self.action_std.fill_(1.0)
            return
        self.action_mean.copy_(actions.mean(dim=0))
        self.action_std.copy_(actions.std(dim=0).clamp_min(1e-6))

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(
            actions.device
        )

    def _denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std.to(actions.device) + self.action_mean.to(
            actions.device
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            tqdm.tqdm.write(message)

    def _log_action_stats(self, actions: torch.Tensor) -> None:
        self._log(
            "[RVQ action stats] "
            f"mean={actions.mean(dim=0).detach().cpu().tolist()} "
            f"std={actions.std(dim=0).detach().cpu().tolist()} "
            f"min={actions.min(dim=0).values.detach().cpu().tolist()} "
            f"max={actions.max(dim=0).values.detach().cpu().tolist()}"
        )

    def _usage_stats_from_counts(self, counts: torch.Tensor) -> dict:
        counts = counts.detach().cpu().float()
        total = counts.sum().clamp_min(1.0)
        probs = counts / total
        nonzero_probs = probs[probs > 0]
        entropy = -(nonzero_probs * torch.log(nonzero_probs)).sum()
        return {
            "hist": counts.long().tolist(),
            "active": int((counts > 0).sum().item()),
            "perplexity": float(torch.exp(entropy).item()),
            "max_usage_ratio": float(probs.max().item()),
        }

    def _code_usage_stats(self, indices: torch.Tensor) -> Tuple[dict, ...]:
        return tuple(
            self._usage_stats_from_counts(
                torch.bincount(
                    indices[:, quantizer_index].detach().reshape(-1).cpu(),
                    minlength=self.codebook_size,
                )
            )
            for quantizer_index in range(self.num_quantizers)
        )

    def _log_usage_stats(self, prefix: str, indices: torch.Tensor) -> None:
        for quantizer_index, stats in enumerate(self._code_usage_stats(indices)):
            self._log(
                f"[RVQ {prefix}] q{quantizer_index} "
                f"hist={stats['hist']} active={stats['active']}/{self.codebook_size} "
                f"perplexity={stats['perplexity']:.2f} "
                f"max_usage_ratio={stats['max_usage_ratio']:.3f}"
            )

    def _print_batch_diagnostics(
        self,
        prefix: str,
        normalized_actions: torch.Tensor,
        normalized_reconstruction: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            actions = self._denormalize_actions(normalized_actions)
            recon = self._denormalize_actions(normalized_reconstruction)
            diff = actions - recon
            self._log(
                f"[RVQ {prefix}] recon_l1={diff.abs().mean().item():.6f} "
                f"recon_mse={diff.pow(2).mean().item():.6f} "
                f"offset_target_abs_mean={diff.abs().mean().item():.6f}"
            )
            self._log_usage_stats(prefix, indices)

    def _print_diagnostics(
        self, prefix: str, normalized_actions: torch.Tensor, max_samples: int = 8192
    ) -> None:
        with torch.no_grad():
            if normalized_actions.shape[0] > max_samples:
                sample_indices = torch.randperm(
                    normalized_actions.shape[0], device=normalized_actions.device
                )[:max_samples]
                normalized_actions = normalized_actions[sample_indices]

            z_e = self.encoder(normalized_actions)
            z_q, indices, _ = self._quantize(z_e)
            normalized_reconstruction = self.decoder(z_q)
            self._print_batch_diagnostics(
                prefix, normalized_actions, normalized_reconstruction, indices
            )

    def _initialize_codebooks_with_residual_kmeans(
        self, normalized_actions: torch.Tensor
    ) -> None:
        with torch.no_grad():
            encoded = self.encoder(normalized_actions)
            residual = encoded
            codebooks = []
            iterator = range(self.num_quantizers)
            if self.verbose:
                iterator = tqdm.tqdm(iterator, desc="RVQ k-means init")
            for _ in iterator:
                centers = self._kmeans(
                    residual,
                    ncluster=self.codebook_size,
                    niter=self.kmeans_iters,
                    verbose=self.verbose,
                )
                codebooks.append(centers)
                nearest = self._nearest_code_indices(residual, centers)
                stats = self._usage_stats_from_counts(
                    torch.bincount(
                        nearest.detach().reshape(-1).cpu(),
                        minlength=self.codebook_size,
                    )
                )
                self._log(
                    f"[RVQ kmeans_init] q{len(codebooks) - 1} "
                    f"hist={stats['hist']} active={stats['active']}/{self.codebook_size} "
                    f"perplexity={stats['perplexity']:.2f} "
                    f"max_usage_ratio={stats['max_usage_ratio']:.3f}"
                )
                residual = residual - centers[nearest]
            self.codebooks.copy_(torch.stack(codebooks, dim=0))

    def _reconstruction_loss(
        self, target: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        if self.reconstruction_loss == "l1":
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)

    @staticmethod
    def _nearest_code_indices(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return torch.argmin(
            torch.sum((x[:, None, :] - codebook[None, :, :]) ** 2, dim=-1),
            dim=-1,
        )

    @staticmethod
    def _kmeans(
        x: torch.Tensor,
        ncluster: int,
        niter: int,
        verbose: bool,
    ) -> torch.Tensor:
        if x.shape[0] < ncluster:
            raise ValueError(
                f"Need at least {ncluster} samples for k-means, got {x.shape[0]}"
            )
        centers = x[torch.randperm(x.shape[0], device=x.device)[:ncluster]].clone()
        iterator = tqdm.trange(niter, disable=not verbose)
        iterator.set_description("RVQ k-means")
        for _ in iterator:
            assignments = ResidualVQVAEActionDiscretizer._nearest_code_indices(
                x, centers
            )
            new_centers = []
            for cluster_idx in range(ncluster):
                members = x[assignments == cluster_idx]
                if members.numel() == 0:
                    replacement = x[
                        torch.randint(x.shape[0], (1,), device=x.device)
                    ].squeeze(0)
                    new_centers.append(replacement)
                else:
                    new_centers.append(members.mean(dim=0))
            centers = torch.stack(new_centers, dim=0)
        return centers

    def _restart_dead_codes(
        self, indices: torch.Tensor, residual_inputs: Tuple[torch.Tensor, ...]
    ) -> None:
        with torch.no_grad():
            for quantizer_index, residual_samples in enumerate(residual_inputs):
                layer_indices = indices[:, quantizer_index].detach()
                counts = torch.bincount(
                    layer_indices.reshape(-1).cpu(), minlength=self.codebook_size
                )
                dead_codes = torch.nonzero(
                    counts <= self.dead_code_threshold, as_tuple=False
                ).flatten()
                if dead_codes.numel() == 0:
                    continue

                selected_codes = F.embedding(
                    layer_indices.to(self.device), self.codebooks[quantizer_index]
                )
                errors = torch.sum(
                    (residual_samples.detach() - selected_codes.detach()) ** 2,
                    dim=-1,
                )
                replacement_order = torch.argsort(errors, descending=True)
                residual_std = residual_samples.detach().std(dim=0).mean().clamp_min(
                    1e-6
                )
                for restart_idx, code_id in enumerate(dead_codes.to(self.device)):
                    sample_idx = replacement_order[
                        restart_idx % replacement_order.shape[0]
                    ]
                    replacement = residual_samples.detach()[sample_idx].clone()
                    if self.restart_noise_scale > 0:
                        replacement = replacement + (
                            torch.randn_like(replacement)
                            * residual_std
                            * self.restart_noise_scale
                        )
                    self.codebooks[quantizer_index, code_id].copy_(replacement)

                self._log(
                    f"[RVQ dead_code_restart] q{quantizer_index} "
                    f"restarted={dead_codes.detach().cpu().tolist()}"
                )

    def _quantize(
        self, z_e: torch.Tensor, return_residuals: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z_e
        quantized_sum = torch.zeros_like(z_e)
        all_indices = []
        residual_inputs = []
        commitment_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
        codebook_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)

        for quantizer_index in range(self.num_quantizers):
            residual_before = residual
            if return_residuals:
                residual_inputs.append(residual_before.detach())
            codebook = self.codebooks[quantizer_index]
            distances = torch.sum(
                (residual_before[:, None, :] - codebook[None, :, :]) ** 2,
                dim=-1,
            )
            indices = torch.argmin(distances, dim=-1)
            selected_codes = F.embedding(indices, codebook)

            commitment_loss = commitment_loss + F.mse_loss(
                residual_before, selected_codes.detach()
            )
            codebook_loss = codebook_loss + F.mse_loss(
                selected_codes, residual_before.detach()
            )
            quantized_sum = quantized_sum + selected_codes
            residual = residual_before - selected_codes.detach()
            all_indices.append(indices)

        z_q = z_e + (quantized_sum - z_e).detach()
        indices = torch.stack(all_indices, dim=-1)
        vq_loss = (
            self.commitment_loss_scale * commitment_loss
            + self.codebook_loss_scale * codebook_loss
        )
        if return_residuals:
            return z_q, indices, vq_loss, tuple(residual_inputs)
        return z_q, indices, vq_loss

    def _codes_to_embeddings(self, flat_codes: torch.Tensor) -> torch.Tensor:
        quantized_sum = torch.zeros(
            flat_codes.shape[0],
            self.embedding_dim,
            device=self.device,
            dtype=self.codebooks.dtype,
        )
        for quantizer_index in range(self.num_quantizers):
            quantized_sum = quantized_sum + F.embedding(
                flat_codes[:, quantizer_index], self.codebooks[quantizer_index]
            )
        return quantized_sum

    @property
    def discretized_space(self) -> int:
        return self.codebook_size

    @property
    def latent_dim(self) -> int:
        return self.num_quantizers

    @property
    def num_latents(self) -> int:
        return self.codebook_size
