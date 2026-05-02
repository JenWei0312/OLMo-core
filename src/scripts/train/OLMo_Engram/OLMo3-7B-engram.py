""" Piggy back on `Olmo3-7B.py` """

from datetime import datetime
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
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
from olmo_core.nn.engram.config import EngramConfig  # <--- for engram
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 2 * 1024
GLOBAL_BATCH_SIZE = 1 * 1024 * 1024  # ~1M tokens


def build_model_config(common: CommonComponents) -> TransformerConfig:
    # ---> DEFINE V1 ENGRAM CONFIG HERE <---
    # (Scale these up from the microscopic test dimensions to real 7B dimensions)

    # 1. We extract the vocab size from AI2's common components FIRST
    vocab_size = common.tokenizer.padded_vocab_size()
    # 2. 
    engram_config = EngramConfig(
        max_ngram_size=3,
        n_embed_per_ngram=1280,
        n_head_per_ngram=8,
        layer_ids=[1, 14],  # early-layer injection and mid-layer injection!
        # Dynamic multiples of the exact tokenizer vocab!
        engram_vocab_size=[vocab_size * 5, vocab_size * 5]
    )
    # --------------------------------------
    # 3. We pass BOTH the vocab size and the engram config to the factory
    return TransformerConfig.olmo3_7B(
        vocab_size=vocab_size,
        engram=engram_config,# <--- THE INJECTION
    )



def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = common.max_sequence_length
    # deleted the if block cause broke

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                # Protect the base vocab AND the Engram memory tables from weight decay
                OptimGroupOverride(
                    params=["embeddings.weight", "*engram_modules*embedding*"], 
                    opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    dataset_config = NumpyFSLDatasetConfig(
        paths=["/workspace/dummy_data/*.npy"], # Hack the path to Point to our single downloaded file!
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        # max target sequence length doesn't affect how the data is loaded, just how it's cached behind the scenes
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10
    # DELETE THESE THREE LINES -- Broke
    #assert common.launch is not None
    #assert len(common.launch.clusters) == 1
    #cluster = common.launch.clusters[0]

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            save_folder=f"./checkpoints/{common.run_name}/", # <-- to be updated once we have compute
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(int(100000)),
            hard_stop=Duration.tokens(int(4e12)),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=20,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                workspace="jenwei0312",                     # <-- to be updated once we have compute
                #save_overwrite=True,                       # <-- nuke this 😭
                project="olmo3-engram-experiments",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )

        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=f"{common.run_name}-engram",          # <-- Differentiate the group
                entity="ai2-llm",
                project="olmo3-engram-experiments",         # <-- Send to an experimental project
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        # no evals for now since we haven't implemented the actual evaluation code for the Engram module yet. Will add in a future iteration once we have something concrete to evaluate on.
        #.with_recommended_evals(common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast")
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
    )
    main(config_builder=config_builder)