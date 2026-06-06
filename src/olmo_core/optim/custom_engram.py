import math
import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict

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
    Explicitly filters 'engram_modules' as a whole out of Dion matrix normalization
    operations and shifts them directly into the non-decayed AdamW fallback loop.
    """
    
    def categorize_parameters(self, model: nn.Module) -> Dict[str, List[str]]:
        assert isinstance(model, Transformer)

        # 1. Standard AI2 Buckets (Keeping their exact logic)
        embed_params = [f"embeddings.{n}" for n, p in model.embeddings.named_parameters() if p.ndim == 2]
        matrix_params = [f"blocks.{n}" for n, p in model.blocks.named_parameters() if p.ndim == 2]
        vector_params = [f"blocks.{n}" for n, p in model.blocks.named_parameters() if p.ndim < 2]
        vector_params += [f"lm_head.{n}" for n, p in model.lm_head.named_parameters() if p.ndim < 2]
        lm_head_params = [f"lm_head.{n}" for n, p in model.lm_head.named_parameters() if p.ndim == 2]

        # 2. OUR ENGRAM BUCKET (Wrapping the whole module up safely!)
        # We use a getattr check just in case you ever run a baseline model without engram
        engram_params = []
        if hasattr(model, "engram_modules"):
            engram_params = [f"engram_modules.{n}" for n, p in model.engram_modules.named_parameters()]

        # 3. Safe 3D Check (Exclude Engram from the strict assertion)
        params_3d_plus = [
            n for n, p in model.named_parameters() 
            if p.ndim > 2 and "engram_modules" not in n
        ]
        assert not params_3d_plus, f"3D+ parameters are not supported outside Engram: {params_3d_plus}"

        # 4. Safe Uncategorized Check (Include our engram_params so it passes!)
        all_model_params = {n for n, p in model.named_parameters() if p.requires_grad}
        categorized_params = set(embed_params + matrix_params + vector_params + lm_head_params + engram_params)
        uncategorized = all_model_params - categorized_params
        assert not uncategorized, f"Uncategorized parameters: {uncategorized}"

        return {
            "embed": embed_params,
            "matrix": matrix_params,
            "vector": vector_params,
            "lm_head": lm_head_params,
            "engram": engram_params,  # Handed cleanly to the overrides
        }

    def default_group_overrides(self, model: nn.Module) -> list[OptimGroupOverride]:
        # Get our safely categorized lists
        params = self.categorize_parameters(model)

        lm_head_out: torch.nn.Linear = model.lm_head.w_out
        model_dim = lm_head_out.weight.shape[1]

        # Standard AI2 Routing
        matrix_override = OptimGroupOverride(params=params["matrix"], opts=dict(algorithm="dion"))
        embed_override = OptimGroupOverride(params=params["embed"], opts=dict(algorithm="adamw", weight_decay=0.0))
        vector_override = OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw"))
        lm_head_override = OptimGroupOverride(
            params=params["lm_head"],
            opts=dict(algorithm="adamw", lr=self.lr / math.sqrt(model_dim)),
        )
        
        # Our Custom Engram Routing (Entire module sent to AdamW with no weight decay)
        engram_override = OptimGroupOverride(
            params=params["engram"], 
            opts=dict(algorithm="adamw", weight_decay=0.0)
        )

        log.info("🎯 Custom Engram routing execution successful! Memory parameters isolated from Dion operations.")
        return [matrix_override, vector_override, embed_override, lm_head_override, engram_override]
    
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