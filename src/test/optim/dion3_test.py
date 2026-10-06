from typing import Any, cast
import pytest
import tempfile
import torch

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.optim.dion import Dion3Config
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.testing import DEVICES, requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_dion
from olmo_core.train import Trainer
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import get_default_device, seed_all
from olmo_core.train.train_module.transformer.common import parallelize_model


class _FakeTrainer:
    """Minimal stand-in for Trainer, just enough for scheduler.set_lr."""

    def __init__(self, global_step: int, max_steps: int):
        self.global_step = global_step
        self.max_steps = max_steps
        self.global_train_tokens_seen = None


def _as_trainer(fake: _FakeTrainer) -> Trainer:
    return cast(Trainer, cast(Any, fake))


# =====================================================================
# 1. Config Verification
# =====================================================================
@requires_dion
def test_dion3_config_builds():
    from dion import Dion3  # type: ignore[reportMissingImports]

    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build()
    optim_cfg = Dion3Config()
    optim = optim_cfg.build(model)

    assert isinstance(optim, Dion3)
    assert len(optim.param_groups) == 4
    for group in optim.param_groups:
        assert "pristine_lr" in group


# =====================================================================
# 2. Local Unit Test: LR Continuity across In-Memory Reset
# =====================================================================
@requires_dion
@pytest.mark.parametrize("device", DEVICES)
def test_dion3_lr_vault_continuity(device: torch.device):
    """
    Verifies that the scheduler vault severs live GPU tensor references
    and correctly prioritizes pristine_lr to avoid double-decay.
    """
    seed_all(0)
    config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)
    model = config.build().train().to(device)
    optim = Dion3Config().build(model)
    scheduler = CosWithWarmup(warmup_steps=5)

    lrs = []
    for step in range(20):
        optim.zero_grad(set_to_none=True)
        x = torch.randint(0, 1024, (2, 8), device=device)
        model(x).sum().backward()
        for group in optim.param_groups:
            lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            lrs.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim.step()

    # The learning rate must follow warmup -> decay without NaN or flatline
    assert lrs[0] < lrs[5]
    assert lrs[5] > lrs[-1]


# =====================================================================
# 3. Distributed Integration Test: HSDP + DCP Resumption
# =====================================================================
def _run_hsdp_train_module_recovery(shard_degree: int, num_replicas: int):
    device = get_default_device()
    seed_all(0)

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        shard_degree=shard_degree,
        num_replicas=num_replicas,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.bfloat16,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
    )
    # Build world mesh ONCE per distributed worker process
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)

    model_config = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2)

    def build_and_wrap():
        model = model_config.build(init_device=device.type).train()
        model = parallelize_model(
            model, world_mesh=world_mesh, device=device, dp_config=dp_config
        )
        optim_cfg = Dion3Config(fraction=0.25, adjust_lr="rms_norm")
        optim = optim_cfg.create_optimizer(model)
        return model, optim

    scheduler = CosWithWarmup(warmup_steps=5)

    # --- Run A: Uninterrupted Baseline ---
    seed_all(0)
    model_a, optim_a = build_and_wrap()
    baseline_lrs = []
    for step in range(20):
        optim_a.zero_grad(set_to_none=True)
        batch = torch.randint(0, 1024, (2, 8), device=device)
        model_a(batch).sum().backward()
        for group in optim_a.param_groups:
            lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            baseline_lrs.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim_a.step()

    # --- Run B: Checkpointed & Resumed Run ---
    seed_all(0)
    model_b, optim_b = build_and_wrap()
    lrs_resumed = []

    for step in range(10):
        optim_b.zero_grad(set_to_none=True)
        batch = torch.randint(0, 1024, (2, 8), device=device)
        model_b(batch).sum().backward()
        for group in optim_b.param_groups:
            lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
            lrs_resumed.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
        optim_b.step()

    # Save & Reload via distributed checkpoint (tests DCP schema matching + pristine_lr)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_model_and_optim_state(tmp_dir, model_b, optim_b)
        load_model_and_optim_state(tmp_dir, model_b, optim_b)

        for step in range(10, 20):
            optim_b.zero_grad(set_to_none=True)
            batch = torch.randint(0, 1024, (2, 8), device=device)
            model_b(batch).sum().backward()
            for group in optim_b.param_groups:
                lr = scheduler.set_lr(group, _as_trainer(_FakeTrainer(step, 20)))
                lrs_resumed.append(lr.item() if isinstance(lr, torch.Tensor) else float(lr))
            optim_b.step()

    # Verify identical trajectories
    assert lrs_resumed == pytest.approx(baseline_lrs), (
        "Resumed learning rate trajectory diverged from uninterrupted run!"
    )

@requires_dion
@requires_multi_gpu
@pytest.mark.parametrize(
    "shard_degree,num_replicas",
    [
        pytest.param(2, 1, id="shard2_replica1"),
        pytest.param(1, 2, id="shard1_replica2"),
    ],
)
def test_hsdp_dion3_module_recovery(shard_degree: int, num_replicas: int):
    seed_all(0)
    run_distributed_test(
        _run_hsdp_train_module_recovery,
        backend="nccl",
        world_size=2,
        func_args=(shard_degree, num_replicas),
    )