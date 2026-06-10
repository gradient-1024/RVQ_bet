import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.action_ae.discretizers.residual_vq import (
    ResidualVQVAEActionDiscretizer,
)
from models.latent_generators.mingpt import MinGPT


def _run_case(
    num_quantizers: int,
    offset_mode: str = "code_independent",
) -> None:
    torch.manual_seed(0)
    batch, seq, action_dim, obs_dim = 32, 4, 2, 6
    angles = torch.linspace(0, 2 * math.pi, 17)[:-1]
    action_centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    action_centers = action_centers * 0.05
    labels = torch.arange(batch * seq) % 16
    actions = action_centers[labels].reshape(batch, seq, action_dim)
    actions = actions + torch.randn_like(actions) * 0.003

    tokenizer = ResidualVQVAEActionDiscretizer(
        action_dim=action_dim,
        num_quantizers=num_quantizers,
        codebook_size=16,
        embedding_dim=action_dim,
        hidden_dim=16,
        hidden_depth=0,
        device="cpu",
        predict_offsets=True,
        train_steps=8,
        batch_size=64,
        learning_rate=1e-3,
        commitment_loss_scale=0.25,
        kmeans_iters=5,
        dead_code_restart_interval=2,
        usage_log_interval=4,
        verbose=True,
    )
    tokenizer.fit_discretizer(actions)

    codes, offsets = tokenizer.encode_into_latent(actions)
    assert codes.shape == (batch, seq, num_quantizers)
    assert offsets.shape == actions.shape
    flat_codes = codes.reshape(-1, num_quantizers)
    usage_stats = tokenizer._code_usage_stats(flat_codes)
    active_counts = [stats["active"] for stats in usage_stats]
    print(f"nq={num_quantizers} active_counts={active_counts}")
    assert min(active_counts) > 1

    decoded_centers = tokenizer.decode_actions(codes)
    assert decoded_centers.shape == actions.shape
    decoded_with_offsets = tokenizer.decode_actions((codes, offsets))
    assert decoded_with_offsets.shape == actions.shape
    assert torch.allclose(decoded_with_offsets, actions, atol=1e-6)

    normalized_actions = tokenizer._normalize_actions(
        actions.reshape(-1, action_dim)
    )
    z_e = tokenizer.encoder(normalized_actions)
    z_q, _, vq_loss = tokenizer._quantize(z_e)
    normalized_recon = tokenizer.decoder(z_q)
    tokenizer_loss = tokenizer._reconstruction_loss(
        normalized_actions, normalized_recon
    ) + vq_loss
    tokenizer.zero_grad(set_to_none=True)
    tokenizer_loss.backward()
    assert any(param.grad is not None for param in tokenizer.parameters())

    policy = MinGPT(
        input_dim=obs_dim,
        n_layer=1,
        n_head=1,
        n_embd=16,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        block_size=seq,
        vocab_size=16,
        latent_dim=tokenizer.latent_dim,
        action_dim=action_dim,
        predict_offsets=True,
        offset_loss_scale=1.0,
        focal_loss_gamma=0.0,
        rvq_num_quantizers=num_quantizers,
        offset_mode=offset_mode,
        code_independent_offsets=offset_mode == "code_independent",
        rvq_code_embedding_dim=tokenizer.embedding_dim,
        code_loss_weights=[1.0] * num_quantizers,
    )
    policy.set_rvq_codebook_embeddings(tokenizer.get_codebook_embeddings())
    obs_rep = torch.randn(batch, seq, obs_dim)
    (logits, pred_offsets), loss, components = policy.get_latent_and_loss(
        obs_rep=obs_rep,
        target_latents=(codes, offsets),
        return_loss_components=True,
    )
    assert logits.shape == (seq, batch, num_quantizers, 16)
    if offset_mode in {"code_independent", "code_embedding_conditioned"}:
        assert pred_offsets.shape == (seq, batch, action_dim)
    else:
        assert pred_offsets.shape == (seq, batch, 16, action_dim)
    if offset_mode == "code_embedding_conditioned":
        assert policy.offset_head[0].in_features == (
            policy.n_embd + num_quantizers * tokenizer.embedding_dim
        )
        selected_embeddings = tokenizer.get_code_embeddings(codes)
        assert selected_embeddings.shape == (
            batch,
            seq,
            num_quantizers,
            tokenizer.embedding_dim,
        )
    assert set(
        [
            "class",
            "acc_code_1",
            "acc3_code_1",
            "offset",
            "offset_target_abs_mean",
            "offset_pred_abs_mean",
            "action_center_abs_error",
            "final_action_recon_mse",
            "total",
        ]
    ).issubset(components.keys())
    if num_quantizers == 2:
        assert set(["acc_code_2", "acc3_code_2", "acc_code_joint"]).issubset(
            components.keys()
        )
    loss.backward()
    assert any(param.grad is not None for param in policy.code_heads.parameters())
    assert any(param.grad is not None for param in policy.offset_head.parameters())

    for latent_sampling_strategy in [None, "sampling", "argmax"]:
        sampled_codes, sampled_offsets = policy.generate_latents(
            obs_rep.transpose(0, 1),
            torch.ones(seq, batch),
            latent_sampling_strategy=latent_sampling_strategy,
        )
        assert sampled_codes.shape == (batch, seq, num_quantizers)
        assert sampled_offsets.shape == (batch, seq, action_dim)
        sampled_actions = tokenizer.decode_actions((sampled_codes, sampled_offsets))
        assert sampled_actions.shape == (batch, seq, action_dim)


def test_rvq_nq1_smoke() -> None:
    _run_case(num_quantizers=1)


def test_rvq_nq1_code_dependent_offset_smoke() -> None:
    _run_case(num_quantizers=1, offset_mode="code_dependent")


def test_rvq_nq2_smoke() -> None:
    _run_case(num_quantizers=2)


def test_rvq_nq2_primary_code_dependent_offset_smoke() -> None:
    _run_case(num_quantizers=2, offset_mode="primary_code_dependent")


def test_rvq_nq2_code_embedding_conditioned_offset_smoke() -> None:
    _run_case(num_quantizers=2, offset_mode="code_embedding_conditioned")


def test_rvq_nq2_rejects_full_code_dependent_offsets() -> None:
    try:
        MinGPT(
            input_dim=6,
            n_layer=1,
            n_head=1,
            n_embd=16,
            block_size=4,
            vocab_size=16,
            action_dim=2,
            predict_offsets=True,
            rvq_num_quantizers=2,
            offset_mode="code_dependent",
            code_independent_offsets=False,
        )
    except ValueError as exc:
        assert "rvq_num_quantizers=1" in str(exc)
    else:
        raise AssertionError("N_q=2 should reject K^2-style code-dependent offsets")


def test_kmeans_argmax_and_sampling_generation_smoke() -> None:
    torch.manual_seed(0)
    batch, seq, action_dim, obs_dim = 8, 3, 2, 6
    policy = MinGPT(
        input_dim=obs_dim,
        n_layer=1,
        n_head=1,
        n_embd=16,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        block_size=seq,
        vocab_size=16,
        action_dim=action_dim,
        predict_offsets=True,
    )
    obs_rep = torch.randn(batch, seq, obs_dim)
    for latent_sampling_strategy in ["sampling", "argmax"]:
        sampled_codes, sampled_offsets = policy.generate_latents(
            obs_rep.transpose(0, 1),
            torch.ones(seq, batch),
            latent_sampling_strategy=latent_sampling_strategy,
        )
        assert sampled_codes.shape == (batch, seq, 1)
        assert sampled_offsets.shape == (batch, seq, action_dim)


if __name__ == "__main__":
    test_rvq_nq1_smoke()
    test_rvq_nq1_code_dependent_offset_smoke()
    test_rvq_nq2_smoke()
    test_rvq_nq2_primary_code_dependent_offset_smoke()
    test_rvq_nq2_code_embedding_conditioned_offset_smoke()
    test_rvq_nq2_rejects_full_code_dependent_offsets()
    test_kmeans_argmax_and_sampling_generation_smoke()
    print("RVQ smoke tests passed")
