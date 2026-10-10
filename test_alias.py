# check_aliasing.py -- throwaway diagnostic, don't commit
import torch

# Same torch 2.7 workaround as in dion3_test.py; must run before any olmo_core import.
if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _orig = torch.compiler.disable

    def _patched(*args, **kwargs):
        kwargs.pop("reason", None)
        return _orig(*args, **kwargs)

    torch.compiler.disable = _patched

import tempfile

from src.olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from src.olmo_core.nn.transformer.config import TransformerConfig
from src.olmo_core.optim.dion import Dion3Config


def build():
    model = TransformerConfig.olmo2_30M(vocab_size=1024, n_layers=2).build().cuda()
    return model, Dion3Config().build(model)


model, optim = build()
print("== after build ==")
for g in optim.param_groups:
    print(type(g["lr"]).__name__, type(g["initial_lr"]).__name__, g["lr"] is g["initial_lr"])

g = optim.param_groups[0]
before = float(g["initial_lr"])
if isinstance(g["lr"], torch.Tensor):
    g["lr"].fill_(0.123)  # what the scheduler does to lr every step
print("initial_lr after in-place lr change:", before, "->", float(g["initial_lr"]))

with tempfile.TemporaryDirectory() as d:
    save_model_and_optim_state(d, model, optim)
    model2, optim2 = build()
    load_model_and_optim_state(d, model2, optim2)
    print("== fresh build, then load ==")
    for i, g in enumerate(optim2.param_groups):
        print(i, "lr:", float(g["lr"]), "initial_lr:", float(g["initial_lr"]))