import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import models.latent_generators.latent_generator as latent_generator

import models.libraries.mingpt.model as mingpt_model
import models.libraries.mingpt.trainer as mingpt_trainer
from models.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Sequence, Tuple


def _build_head(input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> nn.Module:
    if depth <= 0:
        return nn.Linear(input_dim, output_dim)
    layers = []
    current_dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(inplace=True)])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class MinGPT(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        block_size: int = 128,
        vocab_size: int = 50257,
        latent_dim: int = 768,  # Ignore, used for compatibility with other models.
        action_dim: int = 0,
        discrete_input: bool = False,
        predict_offsets: bool = False,
        offset_loss_scale: float = 1.0,
        focal_loss_gamma: float = 2.0,
        rvq_num_quantizers: Optional[int] = None,
        code_loss_weights: Optional[Sequence[float]] = None,
        code_independent_offsets: bool = False,
        rvq_sampling_strategy: str = "argmax",
        rvq_sampling_temperature: float = 1.0,
        rvq_head_hidden_dim: int = 256,
        rvq_head_depth: int = 1,
        rvq_offset_head_hidden_dim: int = 256,
        rvq_offset_head_depth: int = 1,
        offset_mode: Optional[str] = None,
        rvq_code_embedding_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets
        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma
        self.rvq_num_quantizers = rvq_num_quantizers
        self.code_independent_offsets = code_independent_offsets
        self.rvq_sampling_strategy = rvq_sampling_strategy
        self.rvq_sampling_temperature = rvq_sampling_temperature
        self.rvq_head_hidden_dim = rvq_head_hidden_dim
        self.rvq_head_depth = rvq_head_depth
        self.rvq_offset_head_hidden_dim = rvq_offset_head_hidden_dim
        self.rvq_offset_head_depth = rvq_offset_head_depth
        self.offset_mode = self._resolve_offset_mode(
            offset_mode=offset_mode,
            code_independent_offsets=code_independent_offsets,
        )
        self.code_independent_offsets = self.offset_mode == "code_independent"
        self.rvq_code_embedding_dim = (
            action_dim if rvq_code_embedding_dim is None else rvq_code_embedding_dim
        )
        self.use_rvq_heads = rvq_num_quantizers is not None
        if self.use_rvq_heads:
            if self.rvq_num_quantizers is None or self.rvq_num_quantizers < 1:
                raise ValueError("rvq_num_quantizers must be >= 1 when set")
            if code_loss_weights is None:
                self.code_loss_weights = [1.0] * self.rvq_num_quantizers
            else:
                self.code_loss_weights = list(code_loss_weights)
                if len(self.code_loss_weights) != self.rvq_num_quantizers:
                    raise ValueError(
                        "code_loss_weights must match rvq_num_quantizers"
                    )
            if self.predict_offsets:
                self._validate_rvq_offset_mode()
            if self.rvq_sampling_strategy not in {"argmax", "sample"}:
                raise ValueError("rvq_sampling_strategy must be 'argmax' or 'sample'")
            if self.rvq_sampling_temperature <= 0:
                raise ValueError("rvq_sampling_temperature must be > 0")
        for k, v in kwargs.items():
            setattr(self, k, v)

        output_vocab_size = (
            self.vocab_size
            if self.use_rvq_heads
            else (
                self.vocab_size * (1 + self.action_dim)
                if self.predict_offsets
                else self.vocab_size
            )
        )
        gpt_config = mingpt_model.GPTConfig(
            input_size=self.input_size,
            vocab_size=output_vocab_size,
            block_size=self.block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            discrete_input=discrete_input,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        self.model = mingpt_model.GPT(gpt_config)
        if self.use_rvq_heads:
            self.code_heads = nn.ModuleList(
                [
                    _build_head(
                        self.n_embd,
                        self.vocab_size,
                        self.rvq_head_hidden_dim,
                        self.rvq_head_depth,
                    )
                    for _ in range(self.rvq_num_quantizers)
                ]
            )
            if self.predict_offsets:
                offset_input_dim = self.n_embd
                if self.offset_mode == "code_embedding_conditioned":
                    offset_input_dim += (
                        self.rvq_num_quantizers * self.rvq_code_embedding_dim
                    )
                offset_output_dim = self._rvq_offset_output_dim()
                self.offset_head = _build_head(
                    offset_input_dim,
                    offset_output_dim,
                    self.rvq_offset_head_hidden_dim,
                    self.rvq_offset_head_depth,
                )

    @staticmethod
    def _resolve_offset_mode(
        offset_mode: Optional[str],
        code_independent_offsets: bool,
    ) -> str:
        if offset_mode is not None:
            return offset_mode
        return "code_independent" if code_independent_offsets else "code_dependent"

    def _validate_rvq_offset_mode(self) -> None:
        valid_modes = {
            "code_independent",
            "code_dependent",
            "primary_code_dependent",
            "code_embedding_conditioned",
        }
        if self.offset_mode not in valid_modes:
            raise ValueError(f"offset_mode must be one of {sorted(valid_modes)}")
        if self.offset_mode == "code_dependent" and self.rvq_num_quantizers != 1:
            raise ValueError(
                "code_dependent RVQ offsets are only supported for "
                "rvq_num_quantizers=1"
            )
        if self.offset_mode == "primary_code_dependent" and self.rvq_num_quantizers != 2:
            raise ValueError(
                "primary_code_dependent RVQ offsets require rvq_num_quantizers=2"
            )
        if (
            self.offset_mode == "code_embedding_conditioned"
            and self.rvq_num_quantizers != 2
        ):
            raise ValueError(
                "code_embedding_conditioned RVQ offsets require rvq_num_quantizers=2"
            )

    def _get_offset_mode(self) -> str:
        if hasattr(self, "offset_mode"):
            return self.offset_mode
        return self._resolve_offset_mode(
            offset_mode=None,
            code_independent_offsets=getattr(
                self, "code_independent_offsets", False
            ),
        )

    def _rvq_offset_output_dim(self) -> int:
        if self._get_offset_mode() in {"code_dependent", "primary_code_dependent"}:
            return self.vocab_size * self.action_dim
        return self.action_dim

    def set_rvq_codebook_embeddings(self, codebooks: torch.Tensor) -> None:
        codebooks = codebooks.detach().clone()
        if codebooks.ndim != 3:
            raise ValueError(
                "RVQ codebooks must have shape "
                "(rvq_num_quantizers, vocab_size, rvq_code_embedding_dim)"
            )
        expected_shape = (
            self.rvq_num_quantizers,
            self.vocab_size,
            self.rvq_code_embedding_dim,
        )
        if tuple(codebooks.shape) != expected_shape:
            raise ValueError(
                f"Expected RVQ codebooks shape {expected_shape}, got {tuple(codebooks.shape)}"
            )
        self.register_buffer("rvq_codebooks", codebooks)

    def _get_rvq_code_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "rvq_codebooks"):
            raise ValueError(
                "code_embedding_conditioned offsets require RVQ codebooks. "
                "Call set_rvq_codebook_embeddings before training or inference."
            )
        codes = codes.to(self.rvq_codebooks.device).long()
        if codes.shape[-1] != self.rvq_num_quantizers:
            raise ValueError(
                f"Expected {self.rvq_num_quantizers} RVQ codes, got {codes.shape[-1]}"
            )
        embeddings = [
            F.embedding(
                codes[..., quantizer_index],
                self.rvq_codebooks[quantizer_index].detach(),
            )
            for quantizer_index in range(self.rvq_num_quantizers)
        ]
        return torch.cat(embeddings, dim=-1)

    def _select_rvq_offsets(
        self,
        hidden: torch.Tensor,
        codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        offset_mode = self._get_offset_mode()
        if offset_mode == "code_embedding_conditioned":
            code_embeddings = self._get_rvq_code_embeddings(codes).to(hidden.device)
            conditioned_hidden = torch.cat([hidden, code_embeddings], dim=-1)
            selected_offsets = self.offset_head(conditioned_hidden)
            return selected_offsets, selected_offsets

        output_offsets = self.offset_head(hidden)
        if offset_mode == "code_independent":
            return output_offsets, output_offsets

        batch, seq = output_offsets.shape[:2]
        offset_candidates = einops.rearrange(
            output_offsets,
            "batch seq (vocab action) -> (batch seq) vocab action",
            vocab=self.vocab_size,
            action=self.action_dim,
        )
        selected_offsets = offset_candidates[
            torch.arange(offset_candidates.size(0), device=hidden.device),
            codes[..., 0].reshape(-1),
        ].view(batch, seq, self.action_dim)
        return selected_offsets, offset_candidates

    def _format_rvq_offsets_for_output(
        self,
        output_offsets: torch.Tensor,
        batch: int,
        seq: int,
    ) -> torch.Tensor:
        offset_mode = self._get_offset_mode()
        if offset_mode in {"code_independent", "code_embedding_conditioned"}:
            return einops.rearrange(
                output_offsets, "batch seq action -> seq batch action"
            )
        return einops.rearrange(
            output_offsets,
            "(batch seq) vocab action -> seq batch vocab action",
            batch=batch,
            seq=seq,
        )

    @staticmethod
    def _normalize_latent_sampling_strategy(
        sampling_strategy: Optional[str],
    ) -> Optional[str]:
        if sampling_strategy is None:
            return None
        if sampling_strategy == "sampling":
            return "sample"
        if sampling_strategy in {"sample", "argmax"}:
            return sampling_strategy
        raise ValueError(
            "latent_sampling_strategy must be one of: sampling, sample, argmax"
        )

    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unlike torch.transformers, GPT takes in batch x seq_len x embd_dim
        # obs_rep = einops.rearrange(obs_rep, "seq batch embed -> batch seq embed")
        # target_latents = einops.rearrange(
        #     target_latents, "seq batch embed -> batch seq embed"
        # )
        # While this has been trained autoregressively,
        # there is no reason why it needs to be so.
        # We can just use the observation as the input and the next latent as the target.
        if getattr(self, "use_rvq_heads", False):
            return self._get_rvq_latent_and_loss(
                obs_rep=obs_rep,
                target_latents=target_latents,
                return_loss_components=return_loss_components,
            )
        if self.predict_offsets:
            target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )
        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
        if self.predict_offsets:
            output, _ = self.model(obs_rep)
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            batch = logits.shape[0]
            seq = logits.shape[1]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
            # calculate (optionally soft) cross entropy and offset losses
            class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            # offset loss is only calculated on the target class
            # if soft targets, argmax is considered the target class
            selected_offsets = offsets[
                torch.arange(offsets.size(0)),
                target_latents.argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
            ]
            offset_loss = self.offset_loss_scale * F.mse_loss(
                selected_offsets, target_offsets.view(-1, self.action_dim)
            )
            loss = offset_loss + class_loss
            logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
            offsets = einops.rearrange(
                offsets,
                "(N T) V A -> T N V A",  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
                N=batch,
                T=seq,
            )
            if return_loss_components:
                return (
                    (logits, offsets),
                    loss,
                    {"offset": offset_loss, "class": class_loss, "total": loss},
                )
            else:
                return (logits, offsets), loss
        else:
            logits, _ = self.model(obs_rep)
            loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            logits = einops.rearrange(
                logits, "batch seq classes -> seq batch classes"
            )  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            if return_loss_components:
                return logits, loss, {"class": loss, "total": loss}
            else:
                return logits, loss

    def _get_rvq_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_offsets = None
        if type(target_latents) == tuple:
            target_latents, target_offsets = target_latents

        target_latents = target_latents.to(obs_rep.device).long()
        if target_latents.shape[-1] != self.rvq_num_quantizers:
            if self.rvq_num_quantizers == 1:
                target_latents = target_latents.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Expected {self.rvq_num_quantizers} RVQ codes, got {target_latents.shape[-1]}"
                )

        hidden, _ = self.model(obs_rep, return_hidden=True)
        logits_per_codebook = [head(hidden) for head in self.code_heads]
        criterion = FocalLoss(gamma=self.focal_loss_gamma)

        class_loss = torch.zeros((), device=obs_rep.device)
        loss_components = {}
        flat_targets = target_latents.reshape(-1, self.rvq_num_quantizers)
        code_correct_masks = []
        for codebook_index, logits in enumerate(logits_per_codebook):
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_target = flat_targets[:, codebook_index]
            code_loss = criterion(
                flat_logits,
                flat_target,
            )
            weighted_code_loss = (
                self.code_loss_weights[codebook_index] * code_loss
            )
            class_loss = class_loss + weighted_code_loss
            loss_components[f"class_code_{codebook_index}"] = weighted_code_loss
            with torch.no_grad():
                pred_code = flat_logits.argmax(dim=-1)
                code_correct = pred_code == flat_target
                code_correct_masks.append(code_correct)
                code_accuracy = code_correct.float().mean()
                topk = min(3, flat_logits.size(-1))
                code_top3_accuracy = (
                    flat_logits.topk(topk, dim=-1).indices == flat_target[:, None]
                ).any(dim=-1).float().mean()
            loss_components[f"accuracy_code_{codebook_index}"] = code_accuracy
            loss_components[f"acc_code_{codebook_index + 1}"] = code_accuracy
            loss_components[f"acc3_code_{codebook_index + 1}"] = code_top3_accuracy

        loss = class_loss
        loss_components["class"] = class_loss
        if self.rvq_num_quantizers > 0:
            loss_components["code_accuracy"] = torch.stack(
                [
                    loss_components[f"accuracy_code_{codebook_index}"]
                    for codebook_index in range(self.rvq_num_quantizers)
                ]
            ).mean()
        if self.rvq_num_quantizers >= 2:
            loss_components["acc_code_joint"] = torch.stack(
                code_correct_masks, dim=-1
            ).all(dim=-1).float().mean()
        output_offsets = None
        if self.predict_offsets:
            if target_offsets is None:
                raise ValueError("RVQ offset prediction requires target offsets")
            selected_offsets, output_offsets = self._select_rvq_offsets(
                hidden=hidden,
                codes=target_latents,
            )
            raw_offset_mse = F.mse_loss(
                selected_offsets.reshape(-1, self.action_dim),
                target_offsets.to(obs_rep.device).reshape(-1, self.action_dim),
            )
            offset_loss = self.offset_loss_scale * raw_offset_mse
            loss = loss + offset_loss
            loss_components["offset"] = offset_loss
            loss_components["offset_mse"] = raw_offset_mse
            loss_components["offset_target_abs_mean"] = target_offsets.to(
                obs_rep.device
            ).abs().mean()
            loss_components["offset_pred_abs_mean"] = selected_offsets.abs().mean()
            loss_components["action_center_abs_error"] = loss_components[
                "offset_target_abs_mean"
            ]
            loss_components["final_action_recon_mse"] = raw_offset_mse

        output_logits = torch.stack(
            [
                einops.rearrange(logits, "batch seq classes -> seq batch classes")
                for logits in logits_per_codebook
            ],
            dim=2,
        )
        loss_components["total"] = loss
        if output_offsets is not None:
            output_offsets = self._format_rvq_offsets_for_output(
                output_offsets,
                batch=obs_rep.shape[0],
                seq=obs_rep.shape[1],
            )
            output = (output_logits, output_offsets)
        else:
            output = output_logits

        if return_loss_components:
            return output, loss, loss_components
        return output, loss

    def generate_latents(
        self,
        seq_obses: torch.Tensor,
        seq_masks: torch.Tensor,
        latent_sampling_strategy: Optional[str] = None,
    ) -> torch.Tensor:
        seq, batch, embed = seq_obses.size()
        obs_rep = einops.rearrange(seq_obses, "seq batch embed -> batch seq embed")
        requested_sampling_strategy = self._normalize_latent_sampling_strategy(
            latent_sampling_strategy
        )
        if getattr(self, "use_rvq_heads", False):
            hidden, _ = self.model(obs_rep, None, return_hidden=True)
            sampling_strategy = requested_sampling_strategy or getattr(
                self, "rvq_sampling_strategy", "argmax"
            )
            sampling_temperature = getattr(self, "rvq_sampling_temperature", 1.0)
            sampled_codes = []
            for head in self.code_heads:
                logits = head(hidden)
                if sampling_strategy == "argmax":
                    sampled_code = torch.argmax(logits, dim=-1)
                else:
                    probs = F.softmax(logits / sampling_temperature, dim=-1)
                    batch, seq, choices = probs.shape
                    sampled_code = torch.multinomial(
                        probs.reshape(-1, choices), num_samples=1
                    )
                    sampled_code = einops.rearrange(
                        sampled_code,
                        "(batch seq) 1 -> batch seq",
                        batch=batch,
                        seq=seq,
                    )
                sampled_codes.append(sampled_code)
            sampled_codes = torch.stack(sampled_codes, dim=-1)
            if self.predict_offsets:
                sampled_offsets, _ = self._select_rvq_offsets(
                    hidden=hidden,
                    codes=sampled_codes,
                )
                return (sampled_codes, sampled_offsets)
            return sampled_codes
        output, _ = self.model(obs_rep, None)
        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
        else:
            logits = output
        batch, seq, choices = logits.shape
        sampling_strategy = requested_sampling_strategy or "sample"
        if sampling_strategy == "argmax":
            sampled_data = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            # Sample from the multinomial distribution, one per row.
            sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
            sampled_data = einops.rearrange(
                sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
            )
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0], device=offsets.device),
                sampled_data.flatten(),
            ].view(batch, seq, self.action_dim)

            return (sampled_data, sampled_offsets)
        else:
            return sampled_data

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        trainer_cfg = mingpt_trainer.TrainerConfig(
            weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
        )
        optimizer = self.model.configure_optimizers(trainer_cfg)
        if getattr(self, "use_rvq_heads", False):
            optimizer.add_param_group(
                {"params": self.code_heads.parameters(), "weight_decay": weight_decay}
            )
            if self.predict_offsets:
                optimizer.add_param_group(
                    {
                        "params": self.offset_head.parameters(),
                        "weight_decay": weight_decay,
                    }
                )
        return optimizer
