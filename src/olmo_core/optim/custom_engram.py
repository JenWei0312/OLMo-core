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

        embed_params = []
        matrix_params = []
        vector_params = []
        lm_head_params = []
        engram_params = []
        conv_params = [] # added for linear attetion

        # Iterate through every single parameter in the entire model
        for n, p in model.named_parameters():

            
            # 1. OUR ENGRAM BUCKET (Filter this FIRST so it doesn't leak into matrix/vector!)
            if "engram_module" in n:
                engram_params.append(n)

            # 2. CONV1D BUCKET (Catch DGN/Linear Attention 3D weights here!)
            elif "conv1d" in n:
                conv_params.append(n)

            # 3. LM HEAD BUCKETS
            elif "lm_head" in n:
                if p.ndim == 2:
                    lm_head_params.append(n)
                else:
                    vector_params.append(n)
                    
            # 4. EMBEDDING BUCKET
            elif "embeddings" in n and p.ndim == 2:
                embed_params.append(n)
                
            # 5. STANDARD BLOCKS BUCKETS (Attention / FFN)
            elif p.ndim == 2:
                matrix_params.append(n)
            elif p.ndim < 2:
                vector_params.append(n)

        # 5. Safe 3D Check (Exclude Engram from the strict assertion)
        params_3d_plus = [
            n for n, p in model.named_parameters() 
            if p.ndim > 2 and "engram_module" not in n
        ]
        assert not params_3d_plus, f"3D+ parameters are not supported outside Engram: {params_3d_plus}"

        # 6. Safe Uncategorized Check
        all_model_params = {n for n, p in model.named_parameters() if p.requires_grad}
        categorized_params = set(embed_params + matrix_params + vector_params + lm_head_params + engram_params + conv_params)
        uncategorized = all_model_params - categorized_params
        assert not uncategorized, f"Uncategorized parameters: {uncategorized}"

        # Return all 5 keys exactly as default_group_overrides expects them!
        return {
            "matrix": matrix_params,
            "vector": vector_params,
            "embed": embed_params,
            "lm_head": lm_head_params,
            "engram": engram_params,
            "conv": conv_params,     # <-- Return the new bucket
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

        # NEW: Conv1d Routing (Send to AdamW) for linear attenion like GatedDeltaNet
        conv_override = OptimGroupOverride(
            params=params["conv"],
            opts=dict(algorithm="adamw")  # Standard weight decay is usually fine for convs
        )

        log.info("🎯 Custom optimizer routing successful! Engram and Conv1d parameters isolated from Dion operations.")
        return [matrix_override, vector_override, embed_override, lm_head_override, engram_override, conv_override]
    
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