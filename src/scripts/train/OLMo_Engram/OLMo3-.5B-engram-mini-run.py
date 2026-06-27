# ==========================================================================
# 🛑 GANTRY INFRASTRUCTURE EXORCISM (In-Memory Sledgehammer)
# ==========================================================================
import sys
from types import ModuleType

# 1. Create fake in-memory modules to trick AI2's telemetry checks
gantry_mock = ModuleType("gantry")
gantry_callbacks = ModuleType("gantry.callbacks")
gantry_exceptions = ModuleType("gantry.exceptions")
gantry_api = ModuleType("gantry.api")

# 2. Populate the exact stub classes the codebase searches for
gantry_callbacks.Callback = type("Callback", (object,), {})
gantry_exceptions.ExperimentFailedError = type("ExperimentFailedError", (Exception,), {})

class MockGitRepoState:
    @classmethod
    def from_env(cls): return cls()
gantry_api.GitRepoState = MockGitRepoState
gantry_api.Recipe = type("Recipe", (object,), {})

# 3. Force-inject them directly into Python's master runtime module cache
sys.modules["gantry"] = gantry_mock
sys.modules["gantry.callbacks"] = gantry_callbacks
sys.modules["gantry.exceptions"] = gantry_exceptions
sys.modules["gantry.api"] = gantry_api

# ==========================================================================
# 🛑 PYTORCH COMPILER SIGNATURE PATCH (Fix older torch.compiler missing 'reason')
# ==========================================================================
import torch
if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    original_disable = torch.compiler.disable
    def patched_disable(*args, **kwargs):
        # Strip away the 'reason' parameter if it hits an older PyTorch version
        kwargs.pop("reason", None)
        return original_disable(*args, **kwargs)
    torch.compiler.disable = patched_disable

""" Training script for 500M model with Engram with Dion.
5B training tokens.
"""


from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig
)

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
# from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE  #<-- delete cause broke
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.engram.config import EngramConfig  # <--- for engram
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig, CustomEngramDionConfig # < -- Custom engram_dion, clean public import!
from olmo_core.train import Duration, TrainerConfig
from olmo_core.eval import Evaluator # <-- Add Evaluator here
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback, LMEvaluatorCallbackConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig, # <- Add this import for activation checkpointing
)
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode


import numpy as np

# ==========================================
# 0. CONDITIONAL PRE-TRAINING HYPERPARAMETERS (testing vs real training)
# ==========================================

# Sourced from your top-level constant deck
TRAIN_FOR_DEBUG = True  # Set to False when you are ready to remove the network constraints
BASE_MODEL = False
ATTENTION = False

if TRAIN_FOR_DEBUG:
    # Force tight telemetry logging to inspect every single step
    WARMUP_STEPS = 10 # 10 or 20 depending on  integration or debugging run
    METRICS_INTERVAL = 5 # 5 or 10, not necessarily every step, depending on integration or debugging run
    MAX_DURATION = Duration.steps(20) # 20 or 200, depending on integration or debugging run
    EVAL_INTERVAL =  100
    EVAL_ON_FINISH =  True
else:
    print("🚀 INFO: Initializing Full Speed Production Infrastructure...")
    WARMUP_STEPS = 100
    METRICS_INTERVAL = 50
    MAX_DURATION = Duration.tokens(int(5_000_000_000))
    EVAL_INTERVAL = 100
    EVAL_ON_FINISH = True


# ==========================================
# 1. PRIMARY PRE-TRAINING HYPERPARAMETERS (The Source of Truth)
# ==========================================
SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 32 * SEQUENCE_LENGTH  # Token-constant batch size
RANK_MICROBATCH_SIZE = 2 * SEQUENCE_LENGTH  # Sequence size per card

LR = 3e-4
WEIGHT_DECAY = 0.1

# Point directly to your local converted RunPod data paths!
DATA_PATHS = [
    "/workspace/olmo3_data/input_ids_shard_0.npy",
]

# Point directly to your local validation dataset holding path on RunPod NVMe SSD
VAL_DATA_PATH = "/workspace/olmo3_data/input_ids_shard_1.npy"


# ==========================================
# 2. MODEL CONFIGURATION
# ==========================================
def build_model_config(common: CommonComponents) -> TransformerConfig:
    # ‼ seperate base vocalb from engram vocab to avoid wrong LM embedding size
    base_vocab = common.tokenizer.padded_vocab_size()
    # Scale total engram lookup rows dynamically based on official tokenizer size
    engram_vocab = 2 * base_vocab

    if BASE_MODEL:
        return TransformerConfig.olmo3_600M(vocab_size=base_vocab)
    
    # Custom Section 2.2 Sparse Retrieval via Hashed N-grams Mapping
    engram_config = EngramConfig(
        max_ngram_size=3,
        n_embed_per_ngram=1280,
        n_head_per_ngram=8,
        layer_ids=[1, 5],   # Early and mid-layer memory injection
        engram_vocab_size=[engram_vocab, engram_vocab],
    )


    if ATTENTION:
        return TransformerConfig.olmo3_600M(
            vocab_size=base_vocab,
            engram=engram_config,
        )
    
    cfg_gdn_dense = TransformerConfig.olmo3_600M(
            vocab_size=base_vocab,
            engram=engram_config,
        ) # same confid as attention

    # Explicitly tell GDN to use 4 heads so it doesn't crash computing group sizes
    cfg_gdn_dense.block.sequence_mixer = GatedDeltaNetConfig()
    cfg_gdn_dense.block.sequence_mixer.n_heads=10    # set to be 10 or 20, for  "current kernel does not support head dimension larger than 256."

    return cfg_gdn_dense

# ==========================================
# 3. TRAINING ENGINE CONFIGURATION
# ==========================================

def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE ,  
        max_sequence_length=SEQUENCE_LENGTH,       
        # Instantiated with perfect signature validation tracks
        optim=CustomEngramDionConfig(
            lr=3e-4, # bring lr down from default 0.01
            weight_decay=0.1,
        ),
        compile_model=True,  
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16, 
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,  # Prevents Muon / Dion logits from blowing precision caps
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=WARMUP_STEPS),

        #ac_config = TransformerActivationCheckpointingConfig(
        #    mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.85  #<-- activation checkpointing to save memory, set to 85% of available memory
        #)
    )

# ==========================================
# 4. DATALOADER COMPONENTS CONFIGURATION
# ==========================================
def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,  # 🌟 I was delteing in the body, but the argument was missing in the signature line 🤡
) -> DataComponents:
    
    dataset_config =  NumpyFSLDatasetConfig(
        paths=DATA_PATHS,                      # Custom RunPod local NVMe storage paths
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=SEQUENCE_LENGTH,       # Guaranteed alignment matching
        max_target_sequence_length=SEQUENCE_LENGTH, # Single stage fixed length block tuning
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None,           # Dropped messy legacy filter instances
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,   # Track token ceilings safely
        seed=34521, 
        num_workers=4                          # Tuned to optimize RunPod host processor threads
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)



def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10
    
    # Generate an explicit timestamped production run string
    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    # ==========================================================================
    # STAGE A: BLUEPRINT THE VALIDATION INFRASTRUCTURE
    # ==========================================================================
    val_dataset_config =  NumpyPaddedFSLDatasetConfig(
        paths=[VAL_DATA_PATH], 
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=SEQUENCE_LENGTH,            # Pinned to unified source of truth
        # 🗑️ DELETED: max_target_sequence_length
        # 🗑️ DELETED: generate_doc_lengths
        # 🗑️ DELETED: pad_token_id
        metadata=[{"label": None}],                 # Pass as a list if it's a single shard
        instance_filter_config=None,                # Dropped messy legacy filter instances
    )
    
    # 🗑️ NOTE: val_loader_config is DELETED. The callback handles it natively!

    # ==========================================================================
    # STAGE B: ASSEMBLE THE DECLARATIVE TRAINER CONFIGURATION FACTORY
    # ==========================================================================
    return (
        TrainerConfig(
            save_folder=f"/workspace/checkpoints/{common.run_name}/", # Saved safely on local workspace volume
            save_overwrite=True,
            metrics_collect_interval=METRICS_INTERVAL,
            cancel_check_interval=cancel_check_interval,
            # Target limit: 5 Billion Tokens total pre-training window track
            max_duration=MAX_DURATION, 
            hard_stop=Duration.tokens(int(10_000_000_000)),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                # CRITICAL CALIBRATION FOR MINIRUNS: 
                # Since you are using Dion/Muon matrix optimization tracks, the model learns significantly 
                # faster per step than traditional AdamW. Saving a permanent checkpoint every 500 steps 
                # and maintaining an ephemeral sliding snapshot backup every 100 steps ensures you 
                # capture the fast convergence dynamics without burning local disk storage overheads.
                save_interval=200,               
                ephemeral_save_interval=100,     
                enabled=True,                    
                pre_train_checkpoint=False, 
                save_async=True,                 # Offloads heavy disk I/O to background system threads
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                workspace="jenwei0312",                     
                project="olmo3-engram-experiments",
                enabled=False,                              # Explicitly deactivated for this setup
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=f"{common.run_name}-engram",          
                entity="jenwei0312",
                project="olmo3-engram-experiments",         
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )


        # 🌟 THE TRUE AI2 VALIDATION TRACKER 🌟
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset= val_dataset_config,  # <-- Pass eval dataset this way, cleaner
                eval_interval=EVAL_INTERVAL,    # Pause and check validation loss every 100 steps
                eval_on_finish=EVAL_ON_FINISH,  # Guarantee a final eval when the 5B tokens are done
                eval_duration=Duration.steps(20),
                log_interval=5,
            ),
        )

    )


if __name__ == "__main__":
    import sys
    
    # Force the exact action arguments that olmo_core's internal parser needs!
    # [0] is the script path, [1] is the subcommand, [2] is a dummy run name, [3] is the cluster type
    sys.argv = [sys.argv[0], "train", "olmo3-500m-engram-run", "local"]

    # Re-import the native experiment runner safely
    from olmo_core.internal.experiment import main

    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=False,  
    )
    
    print("🚀 Bootstrapping framework orchestrator via localized profile injection...")
    main(config_builder=config_builder)