import math
import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

from olmo_core.optim.config import OptimGroupOverride
from olmo_core.nn.transformer import Transformer
from olmo_core.optim.config import OptimConfig
from olmo_core.optim.dion import DionConfig

log = logging.getLogger(__name__)

# Register it with an explicit string identifier under the DionConfig family
@OptimConfig.register("custom_engram_dion")
@dataclass
class CustomEngramDionConfig(DionConfig):
    """
    A custom optimizer blueprint that extends DionConfig.
    Explicitly filters 'engram_modules' variables out of Dion matrix normalization
    operations and shifts them directly into the non-decayed AdamW fallback loop.
    """
    
    def default_group_overrides(self, model: torch.nn.Module) -> List[OptimGroupOverride]:
        assert isinstance(model, Transformer)
        
        # 1. Gather default layer classifications via native lookups
        params = self.categorize_parameters(model)
        
        # 2. Extract any custom Engram tensors captured in the matrix track
        engram_matrix_params = [p for p in params["matrix"] if "engram_modules" in p]
        
        # 3. Clear those parameters out of the core Dion matrix array
        params["matrix"] = [p for p in params["matrix"] if "engram_modules" not in p]
        
        # 4. Route them straight into the embed group (AdamW fallback track)
        params["embed"].extend(engram_matrix_params)
        
        # 5. Extract structural hidden dimensions for the lm_head multiplier math step
        lm_head_out: torch.nn.Linear = model.lm_head.w_out
        model_dim = lm_head_out.weight.shape[1]

        # 6. Assemble your priority-resolved parameter execution groups
        matrix_override = OptimGroupOverride(params=params["matrix"], opts=dict(algorithm="dion"))
        vector_override = OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw"))
        
        # Your custom multi-head embedding slots are now safely bounded here with weight_decay = 0.0
        embed_override = OptimGroupOverride(params=params["embed"], opts=dict(algorithm="adamw", weight_decay=0.0))
        
        lm_head_override = OptimGroupOverride(
            params=params["lm_head"],
            opts=dict(algorithm="adamw", lr=self.lr / math.sqrt(model_dim)),
        )

        log.info("🎯 Custom Engram routing execution successful! Memory parameters isolated from Dion operations.")
        return [matrix_override, vector_override, embed_override, lm_head_override]
    
    def create_optimizer(self, model: torch.nn.Module, strict: bool = True, **kwargs):
    # When using Dion, we need to set the recompile limit to 16 to avoid triggering an error
    # due to too many recompile requests. Typically, on the second recompilation, torch attempts
    # to compile a dynamic version of the op, unless dynamic=False is marked. Too many different
    # shapes passed to a compiled op with dynamic=False will trigger this error. Since we have
    # grad matrices with many different shapes, we need to set the recompile limit higher than
    # the default of 8.
    # https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html
        #torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16) <-- fix .recompile_limit error

        parallelism_config = self.build_parallelism_config()
        optim = self.optimizer()(
            self.build_groups(model, strict=strict),
            replicate_mesh_grad_sync=False,  # HSDP / FSDP / DDP will handle gradient sync internally
            **parallelism_config,
            **kwargs,
        )
        return optim