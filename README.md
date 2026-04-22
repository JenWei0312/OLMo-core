<div align="center">
  <img src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/olmo2/olmo.png" alt="OLMo Logo" width="280" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <h1>OLMo-core + Engram</h1>
  <h4>Integrating DeepSeek's Conditional Memory Module into OLMo-core</h4>
  <p><em>An independent research integration by <a href="https://www.linkedin.com/in/YOUR_LINKEDIN">Jen Wei</a></em></p>
</div>

> **Status:** Active development. Forward pass and autograd verified ✅. Training runs pending compute access. See [Roadmap](#roadmap) for full scope.

---

## What This Is

This fork integrates [DeepSeek's Engram module](https://github.com/deepseek-ai/Engram) — a conditional memory mechanism based on n-gram lookup — into [AI2's OLMo-core](https://github.com/allenai/OLMo-core) training infrastructure as an optional architectural component.

Engram provides O(1) static knowledge retrieval as a complementary sparsity axis alongside FFN and MoE computation. This integration makes Engram available to any OLMo-core model configuration via a single config flag, with verified gradient flow through the full training stack.

This is not an official AI2 project. It is independent research motivated by the architectural questions below.

---

## Why This Is Interesting

### In-Context vs. Out-of-Context Memory

OLMo Hybrid's GatedDeltaNet layers perform *in-context* associative memory — dynamically updating recurrent state as they process a sequence. Standard attention similarly operates within the context window. Both mechanisms are fundamentally bounded by sequence length and hidden state capacity.

Engram operates on a qualitatively different axis: *out-of-context* static memory. Its n-gram embedding table is built from the training corpus offline and retrieved via deterministic hash lookup at inference time. It does not learn from the current sequence — it recognizes patterns it has seen before.

These are not competing mechanisms. They address different failure modes:
- Attention and linear RNNs handle *what has this sequence taught me so far*
- Engram handles *what do I already know about this token pattern from pretraining*

The hypothesis is that Engram frees up hidden state capacity in GDN and attention layers for genuinely dynamic, context-specific associations — because static pattern reconstruction is offloaded to the lookup table.

### Parameter Allocation and the U-Shaped Scaling Law

DeepSeek's paper identifies an optimal allocation of ~20-25% of sparse model capacity to Engram and ~75-80% to dynamic computation (MoE). Pure MoE is suboptimal because parameter budget is wasted reconstructing static patterns that a lookup table could retrieve in O(1).

The more rigorous question is not *does Engram help* — more parameters almost always help — but *is Engram the most efficient use of that parameter budget?* The U-shaped scaling law suggests yes, up to the optimal allocation point.

This question becomes more interesting in hybrid architectures where GDN layers are already parameter-lean. The optimal allocation curve may shift when the sequence mixer has finite recurrent state capacity.

### Numerical Stability as a Bonus Property

Embedding tables are numerically boring in the best possible way — no nonlinearities, no gradient flow through complex operations, no activation functions to saturate. This makes them:

- Quantization-friendly (INT8/INT4 with minimal quality degradation)
- Training-stable by construction, independent of any other stability mechanisms
- A natural complement to architecturally complex components like MoE routing or GDN delta updates

Trading some FFN/MoE parameter budget for embedding table budget is therefore not just a compute efficiency argument — it is also a numerical stability argument. This property holds regardless of what other stability mechanisms (e.g. DeepSeek's manifold hyper-connections) are present in the architecture.

### Inference Efficiency

Engram's embedding table can be offloaded to CPU DRAM with sub-3% throughput penalty via asynchronous PCIe retrieval, masked by early-layer GPU computation. This decouples knowledge storage from GPU HBM — a meaningful practical advantage for deployment at scale.

**Note:** CPU offloading is a known future optimization in this integration. The current implementation runs the embedding table on GPU. See [Roadmap](#roadmap).

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| `EngramConfig` dataclass | ✅ Done | Integrated into OLMo-core config system |
| `Engram` module | ✅ Done | Ported and de-hyper-connected from DeepSeek reference |
| `CompressedTokenizer` | ✅ Done | Vocab compression via canonical normalization |
| `NgramHashMapping` | ✅ Done | Multi-head hashing, vectorized across heads |
| `MultiHeadEmbedding` | ✅ Done | Single embedding table with offset indexing |
| `ShortConv` | ✅ Done | Depthwise conv for local context fusion |
| Injection into `Transformer.forward()` | ✅ Done | Pre-block residual addition, configurable layer IDs |
| Autograd / backward pass | ✅ Done | Full gradient flow verified on CUDA |
| GDN (OLMo Hybrid) layer integration | 🚧 Next | Pure attention skeleton tested; hybrid layers pending |
| Training script | 🚧 Pending | Blocked on compute access |
| CPU DRAM offloading | 🔮 Future | Known bottleneck; requires async PCIe engineering |
| ROCm / AMD validation | 🔮 Future | Planned; motivated by hardware accessibility |

---

## Experimental Design

The core research question is: **how does Engram interact with different combinations of sequence mixer and FFN component?**

This motivates a 2×2 ablation across OLMo-core's existing model families:

|  | **Dense FFN** | **MoE** |
|--|---------------|---------|
| **Attention** | OLMo-2 / OLMo-3 | OLMoE |
| **Linear RNN (GDN)** | OLMo Hybrid + FFN | OLMo Hybrid + MoE |

**Primary hypothesis:** Engram provides the largest relative gain in the Attention + Dense FFN cell, because dense FFN has no existing sparsity mechanism and every token pays the full FFN compute cost regardless of whether it needs it. Engram's static pattern offloading should show the clearest signal here.

**Secondary hypothesis:** In the Linear RNN cells, Engram's benefit is partially mediated by freeing GDN recurrent state capacity for dynamic associations — a qualitatively different mechanism than in pure attention models.

**Priority for initial training run:** Attention + Dense FFN (OLMo-3 7B + ~1B Engram). Cleanest baseline, fewest confounds, most interpretable results.

> 💡 **Compute needed:** A minimal signal run (~10B tokens on 8B parameter model) requires approximately 3-10 days on 8×A100s. If you have access to compute and find this work interesting, please reach out.

---

## Roadmap

**Phase 1 — Integration (current)**
- [x] Engram module ported and integrated into OLMo-core
- [x] Forward and backward pass verified
- [ ] OLMo Hybrid (GDN) layer compatibility
- [ ] Clean training script for OLMo-3 7B + Engram

**Phase 2 — Minimal Training Run**
- [ ] 10-50B token run on Attention + Dense FFN config
- [ ] Loss curve analysis and perplexity comparison vs baseline
- [ ] Parameter allocation ablation (vary Engram size vs FFN budget)

**Phase 3 — Broader Ablations**
- [ ] 2×2 grid experiments across model families
- [ ] Layer placement ablation (layer 2 vs layer 4 injection)
- [ ] Long-context evaluation (Engram's structural advantage)

**Phase 4 — Efficiency**
- [ ] GPU-native hash computation (remove CPU bottleneck)
- [ ] CPU DRAM offloading for embedding table
- [ ] ROCm / AMD hardware validation

---

## Quick Start

```bash
# Clone this fork
git clone -b feature/engram-poc https://github.com/JenWei0312/OLMo-core.git
cd OLMo-core

# Install
pip install -e .
pip install sympy tokenizers transformers

# Run the integration test
python src/integration_tests/test_engram_forward.py
```

Expected output: forward pass, loss calculation, backward pass, and optimizer step all succeed on CUDA.

---

## Citation

If you find this integration useful, please also cite the original Engram paper:

```bibtex
@article{engram2025,
  title={Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2601.07372},
  year={2025}
}
```

And the OLMo-core paper:

```bibtex
@misc{olmo20242olmo2furious,
      title={{2 OLMo 2 Furious}},
      author={{Team OLMo} and Pete Walsh and Luca Soldaini and others},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
}
```

---

*This is independent research conducted without institutional affiliation or compute resources. Feedback and collaboration welcome.*

---

## Original OLMo-core Documentation

> Everything below is the original README from [allenai/OLMo-core](https://github.com/allenai/OLMo-core).


<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/olmo2/olmo.png" alt="OLMo Logo" width="280" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <h1>OLMo-core</h1>
  <h4>Building blocks for OLMo modeling and training</h4>
</div>
<p align="center">
  <a href="https://olmo-core.readthedocs.io/en/latest/">
    <img alt="Docs" src="https://img.shields.io/badge/API-docs-red">
  </a>
  <a href="https://github.com/allenai/OLMo-core/tree/main/src/examples">
    <img alt="Examples" src="https://img.shields.io/badge/API-examples-994B00">
  </a>
  <a href="https://github.com/allenai/OLMo-core/releases/tag/v1.9.0">
    <img alt="Pypi" src="https://img.shields.io/pypi/v/ai2-olmo-core.svg">
  </a>
  <a href="https://github.com/allenai/OLMo-core/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://arxiv.org/pdf/2501.00656.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-orange">
  </a>
  <a href="https://playground.allenai.org">
    <img alt="Playground" src="https://img.shields.io/badge/Ai2-Playground-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware.

For development, we recommend installing from source:

```bash
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```
Or you can install from PyPI with:

```bash
pip install ai2-olmo-core
```

There are a number of optional dependencies that must be installed to use certain functionality as well, including:

- [flash-attn](https://github.com/Dao-AILab/flash-attention), [ring-flash-attn](https://github.com/zhuzilin/ring-flash-attention), and [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) for the corresponding attention backends.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) for a low-memory "fused-linear" loss implementation.
- [torchao](https://github.com/pytorch/ao) for float8 training.
- [grouped_gemm](https://github.com/tgale96/grouped_gemm) for dropless mixture-of-experts (MoE) models. You may need to compile from source until [PR #21](https://github.com/tgale96/grouped_gemm/pull/21) is released (post v0.1.6).
- [QuACK](https://github.com/Dao-AILab/quack) for some CuTe-based kernels.

The published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies, and are regularly tested on our in-house H100 clusters.
But there are several things to keep in mind if you intend to use these images:

- They do not come with the OLMo-core package installed, only its dependencies, to accommodate for regular code changes.
- They may not work on your own cluster if you have different hardware or driver/CUDA versions.

If the published images do not work for your use-case for any of the above reasons, you could adapt our [Dockerfile](https://github.com/allenai/OLMo-core/blob/main/src/Dockerfile) to build your own images.

## Official training scripts

Official training scripts for released models can be found in [`src/scripts/official/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official).

These scripts are meant to be launched with ``torchrun``, or with OLMo-core's Beaker launch CLI if you have access to Beaker.

For example:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints
```

You can override most configuration options from the command-line. For example, to override the learning rate you could launch the script like this:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-train.py \
  --save-folder=/path/to/save/checkpoints \
  --train_module.optim.lr=6e-3
```

To continue annealing from a checkpoint, we use a separate script which can be launched like this:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo2/OLMo-2-0325-32B-anneal.py \
  --save-folder=/path/to/save/checkpoints \
  --checkpoint=https://olmo-checkpoints.org/ai2-llm/peteish32/step721901
```

### Available Training Scripts

| Model Family | Directory | Description |
|--------------|-----------|-------------|
| **OLMo-2** | [`src/scripts/official/OLMo2/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo2) | Training scripts and model card for OLMo-2 32B models |
| **OLMo-3** | [`src/scripts/official/OLMo3/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3) | Training scripts and model cards for OLMo-3 7B and 32B models |

## Inference

### With Hugging Face Transformers

You can use our Hugging Face [transformers](https://github.com/huggingface/transformers) integration to run inference on the OLMo checkpoints:

```bash
pip install transformers>=4.57.0
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1125-32B")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1125-32B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# inputs = {k: v.to('cuda') for k,v in inputs.items()} # optional verifying cuda
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=0.7)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/Olmo-3-1125-32B")
print(olmo_pipe("Language modeling is"))
```

### With vLLM

[vLLM](https://docs.vllm.ai/en/latest/) provides high-throughput inference for OLMo models. You can use it for offline batched inference:

```bash
pip install vllm>=0.11.0
```

```python
from vllm import LLM, SamplingParams
llm = LLM(model="allenai/Olmo-3-1125-32B")
sampling_params = SamplingParams(temperature=1.0, top_p=0.7)
prompts = ["Language modeling is"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

For more details, see the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference).

### With Olmo-core (beta)

Autoregressive generation is supported directly in Olmo-core. Using this capability, we provide a chat-loop demo that can be used to interact with models in an interactive chat session:

```bash
python -m olmo_core.generate.chat https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage3/step11921/ --max-new-tokens 512
```

## Evaluation

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/OLMo-eval) and [olmes](https://github.com/allenai/olmes) repositories.

## Development

The Python library source code is located in `src/olmo_core`. The corresponding tests are located in `src/test`. The library docs are located in `docs`. You can build the docs locally with `make docs`.

Code checks:

- We use `pytest` to run tests. You can run all tests with `pytest -v src/test`. You can also point `pytest` at a specific test file to run it individually.
- We use `isort` and `black` for code formatting. Ideally you should integrate these into your editor, but you can also run them manually or configure them with a pre-commit hook. To validate that all files are formatted correctly, run `make style-check`.
- We use `ruff` as our primary linter. You can run it with `make lint-check`.
- We use `mypy` as our type checker. You can run it with `make type-check`.

## Citing

```bibtex
@misc{olmo20242olmo2furious,
      title={{2 OLMo 2 Furious}},
      author={{Team OLMo} and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656},
}
```
