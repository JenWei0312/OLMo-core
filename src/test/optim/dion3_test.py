from typing import Any, cast
import pytest
import torch

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.dion import Dion3Config
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.testing import DEVICES, requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_dion
from olmo_core.train import Trainer
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all


def build_transformer_model() -> Transformer:
    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build()
    return model


class _FakeTrainer:
    """Minimal stand-in for Trainer, just enough for scheduler.set_lr."""

    def __init__(self, global_step: int, max_steps: int):
        self.global_step = global_step
        self.max_steps = max_steps
        self.global_train_tokens_seen = None


def _as_trainer(fake: _FakeTrainer) -> Trainer:
    """Cast helper to silence static type checker (Pylance)."""
    return cast(Trainer, cast(Any, fake))


# =====================================================================
# 1. Config Building Test
# =====================================================================
@requires_dion
def test_dion3_config_to_optim():
    from dion import Dion3  # type: ignore[reportMissingImports]

    config = Dion3Config()
    model = build_transformer_model()
    optim = config.build(model)

    assert isinstance(optim, Dion3)
    assert len(optim.param_groups) == 4  # emb, matrix, vector, lm_head


# =====================================================================
# 2. Local Unit Test: LR Continuity across Checkpoint Load
# =====================================================================
@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_dion3_lr_survives_checkpoint_recovery(device: torch.device, tmp_path):
    seed_all(0)
    config = Dion3Config()
    model = build_transformer_model().train().to(device)
    optim = config.build(model)
    scheduler = CosWithWarmup(warmup_steps=5)

    def train_steps(start_step: int, end_step: int, max_steps: int):
        lrs = []
        for step in range(start_step, end_step):
            optim.zero_grad(set_to_none=True)
            model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
            for group in optim.param_groups:
                lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, max_steps)))
                lrs.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
            optim.step()
        return lrs

    # Uninterrupted baseline
    seed_all(0)
    baseline_lrs = train_steps(0, 20, max_steps=20)

    # Interrupted run with checkpoint save & restore
    seed_all(0)
    model2 = build_transformer_model().train().to(device)
    optim2 = config.build(model2)
    scheduler2 = CosWithWarmup(warmup_steps=5)

    lrs_before = []
    for step in range(10):
        optim2.zero_grad(set_to_none=True)
        model2(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
        for group in optim2.param_groups:
            lr = scheduler2.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            lrs_before.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim2.step()

    save_model_and_optim_state(tmp_path, model2, optim2)
    load_model_and_optim_state(tmp_path, model2, optim2)

    lrs_after = []
    for step in range(10, 20):
        optim2.zero_grad(set_to_none=True)
        model2(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
        for group in optim2.param_groups:
            lr = scheduler2.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            lrs_after.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim2.step()

    recovered_lrs = lrs_before + lrs_after
    assert recovered_lrs == pytest.approx(baseline_lrs), (
        "LR trajectory diverged after checkpoint recovery!"
    )


# =====================================================================
# 3. Distributed Integration Test: HSDP + DCP Resumption
# =====================================================================
def _run_hsdp_dion3_recovery(shard_degree: int, num_replicas: int):
    device = get_default_device()

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp, shard_degree=shard_degree, num_replicas=num_replicas
    )
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)

    def build_and_wrap():
        model = config.build(init_device=device.type).train()
        model = parallelize_model(
            model, world_mesh=world_mesh, device=device, dp_config=dp_config
        )
        optim = Dion3Config().create_optimizer(model)
        return model, optim

    scheduler = CosWithWarmup(warmup_steps=5)

    # 1. Baseline Run
    seed_all(0)
    model, optim = build_and_wrap()
    baseline_lrs = []
    for step in range(20):
        optim.zero_grad(set_to_none=True)
        model(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
        for group in optim.param_groups:
            lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            baseline_lrs.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim.step()

    # 2. Checkpointed Run
    seed_all(0)
    model2, optim2 = build_and_wrap()
    lrs_before = []
    for step in range(10):
        optim2.zero_grad(set_to_none=True)
        model2(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
        for group in optim2.param_groups:
            lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            lrs_before.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim2.step()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        save_model_and_optim_state(tmp_path, model2, optim2)
        load_model_and_optim_state(tmp_path, model2, optim2)

        lrs_after = []
        for step in range(10, 20):
            optim2.zero_grad(set_to_none=True)
            model2(torch.randint(0, 1024, (2, 8), device=device).int()).sum().backward()
            for group in optim2.param_groups:
                lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
                lrs_after.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
            optim2.step()

    recovered_lrs = lrs_before + lrs_after
    assert recovered_lrs == pytest.approx(baseline_lrs)


@requires_dion
@requires_multi_gpu
@pytest.mark.parametrize(
    "shard_degree,num_replicas",
    [
        pytest.param(2, 1, id="shard2_replica1"),
        pytest.param(1, 2, id="shard1_replica2"),
    ],
)
def test_hsdp_dion3_recovery(shard_degree: int, num_replicas: int):
    seed_all(0)
    run_distributed_test(
        _run_hsdp_dion3_recovery,
        backend="nccl",
        world_size=2,
        func_args=(shard_degree, num_replicas),
    )