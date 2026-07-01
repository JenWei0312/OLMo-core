#!/bin/bash
set -e  # Terminate execution if any step fails

echo "🚀 Starting Ephemeral Environment Setup..."

# ==========================================
# 1. ISOLATED PYTHON ECOSYSTEM
# ==========================================
if [ ! -d "/workspace/my_env" ]; then
    echo "📦 Creating virtual environment at /workspace/my_env..."
    python -m venv /workspace/my_env
else
    echo "✅ Virtual environment already exists."
fi

source /workspace/my_env/bin/activate

# ==========================================
# 2. FRAMEWORK & OPTIMIZER PRIMITIVES
# ==========================================
echo "⚙️ Installing OLMo-core dependencies..."
pip install -e .[all]

echo "⚡ Integrating Orthonormal Optimizer Subspaces (Dion)..."
# Compiles the dynamic Triton NS matrices natively from Microsoft
pip install git+https://github.com/microsoft/dion.git --no-cache-dir

# ==========================================
# 3. PYTORCH ARCHITECTURE LOCKDOWN
# ==========================================
# We completely override and drop down to a stable version with fully supported 
# torch.compiler.disable(reason=...) keyword arguments.
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "None")
if [[ "$CURRENT_TORCH" != *"2.6.0"* ]]; then
    echo "🔥 Purging old torch variants and installing production-stable PyTorch 2.6.0..."
    pip uninstall -y torch torchvision torchaudio
    # Installs the precise production tracking line compatible with modern olmo-core structures
    pip install torch==2.6.0+cu124 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
else
    echo "✅ PyTorch version verified as clean."
fi

# ==========================================
# 3b. FLASH-LINEAR-ATTENTION (fla)
# ==========================================
# Installed AFTER the torch lockdown, and with --no-deps, so fla can't drag in its own torch/triton and silently override the pinned torch 2.6.0 above.
# Fixes the known "arith.mulf op requires the same encoding for all operands and results" MLIR error in fla/ops/gated_delta_rule/wy_fast.py, which is a
# fla kernel bug (patched upstream) — NOT a torch version issue. No need to downgrade torch to chase this one.
echo "👀 Installing latest fla (no-deps to preserve torch 2.6.0 lock)..."
pip install -U flash-linear-attention --no-deps

# Sanity check: confirm which triton is actually active after the fla install. 
# torch 2.6.0 ships with its own bundled pytorch-triton (~3.1.x/3.2.x), which should be compatible. If the MLIR error still appears after this, the next
# lever to pull is an explicit triton pin, e.g.: pip install -U triton
ACTIVE_TRITON=$(python -c "import triton; print(triton.__version__)" 2>/dev/null || echo "None")
echo "🔍 Active triton version: ${ACTIVE_TRITON}"

# ==========================================
# 4. THE EXORCISM & PHANTOM BYPASSES
# ==========================================
echo "🧟 Exorcising Zombie Libraries..."
pip uninstall -y flash-attn cutlass torch_c_dlpack_ext || true

echo "👻 Setting up Phantom Imports..."
# Find the active venv site-packages directory
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')

# --- Flash Attention Fake ---
mkdir -p $SITE_PACKAGES/flash_attn/cute
touch $SITE_PACKAGES/flash_attn/__init__.py
echo "class Dummy: pass" > $SITE_PACKAGES/flash_attn/cute/__init__.py
echo "class Dummy: pass" > $SITE_PACKAGES/flash_attn/cute/interface.py

# --- Gantry/Beaker Fake ---
mkdir -p $SITE_PACKAGES/gantry
touch $SITE_PACKAGES/gantry/__init__.py
echo "class Callback: pass" > $SITE_PACKAGES/gantry/callbacks.py
echo "class ExperimentFailedError(Exception): pass" > $SITE_PACKAGES/gantry/exceptions.py
cat << 'EOF' > $SITE_PACKAGES/gantry/api.py
class GitRepoState:
    @classmethod
    def from_env(cls):
        return cls()
class Recipe: pass
EOF

# --- Sledgehammer Core File Scrubber ---
echo "🔨 Applying Sledgehammer sed Bypasses..."
find src -type f -name "*.py" -exec sed -i 's/AttentionBackendName\.flash_2/AttentionBackendName\.torch/g' {} +
find src -type f -name "*.py" -exec sed -i 's/"flash_2"/"torch"/g' {} +
find src -type f -name "*.py" -exec sed -i "s/'flash_2'/'torch'/g" {} +
find src -type f -name "*.py" -exec sed -i 's/m\.apply_compile()/pass/g' {} +
sed -i 's/.*@GantryCallback.register.*/# Bypassed Beaker/g' src/olmo_core/launch/beaker.py || true

# ==========================================
# 5. GOOGLE DRIVE DATA SYNCHRONIZATION & CONVERSION
# ==========================================
echo "📊 Fetching Real Mid-Training Shards from Remote Storage..."
mkdir -p /workspace/olmo3_data

# Ensure gdown and numpy are available in the venv
pip install gdown numpy --quiet

# Download Shard 0 as raw binary
if [ ! -f "/workspace/olmo3_data/input_ids_shard_0.npy" ]; then
    echo "📥 Syncing Shard Pool 0..."
    # REMOVED --id flag: Pass the ID directly as a positional argument
    gdown "1klEBeFjonNiCYyepaglUwLOZBudxJiIq" -O /workspace/olmo3_data/input_ids_shard_0.bin
fi

# Download Shard 1 as raw binary
if [ ! -f "/workspace/olmo3_data/input_ids_shard_1.npy" ]; then
    echo "📥 Syncing Shard Pool 1..."
    # REMOVED --id flag: Pass the ID directly as a positional argument
    gdown "1UguTXOC6tYezU06Jr0xsxbZxzXS99reN" -O /workspace/olmo3_data/input_ids_shard_1.bin
fi

# Convert raw .bin to memory-mapped .npy automatically
echo "🔄 Formatting binary streams into PyTorch-ready .npy matrices..."
python -c "
import numpy as np
import os

for i in range(2):
    bin_path = f'/workspace/olmo3_data/input_ids_shard_{i}.bin'
    npy_path = f'/workspace/olmo3_data/input_ids_shard_{i}.npy'

    if os.path.exists(bin_path) and not os.path.exists(npy_path):
        print(f'  -> Converting Shard {i}...')
        raw_data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        np.save(npy_path, raw_data)
        os.remove(bin_path)
        print(f'  -> Shard {i} complete and cleaned.')
"