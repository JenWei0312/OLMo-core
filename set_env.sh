#!/bin/bash

echo "🚀 Starting Environment Setup..."

# 1. Create the virtual environment ONLY if it doesn't exist
if [ ! -d "/workspace/my_env" ]; then
    echo "📦 Creating new virtual environment at /workspace/my_env..."
    python -m venv /workspace/my_env
else
    echo "✅ Virtual environment already exists."
fi

# 2. Activate it
source /workspace/my_env/bin/activate

# Quick check to ensure activation worked
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🟢 Activated: $VIRTUAL_ENV"
else
    echo "🔴 Failed to activate virtual environment!"
    return 1
fi

# 3. Base Installation
echo "⚙️ Installing OLMo-core dependencies..."
pip install -e .[all]

# 4. The PyTorch Sync (Only install if we don't have 2.6.0)
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$CURRENT_TORCH" != *"2.6.0"* ]]; then
    echo "🔥 Syncing PyTorch 2.6.0 (Currently $CURRENT_TORCH)..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
else
    echo "✅ PyTorch 2.6.0 is already installed."
fi

# 5. The Exorcism
echo "🧟 Exorcising Zombie Libraries..."
pip uninstall -y flash-attn cutlass torch_c_dlpack_ext

# 6. The Phantom Bypass
echo "👻 Setting up Phantom Imports..."
mkdir -p flash_attn/cute
touch flash_attn/__init__.py
echo "class Dummy: pass" > flash_attn/cute/__init__.py
echo "class Dummy: pass" > flash_attn/cute/interface.py

# 7. The Sledgehammer Bypasses
echo "🔨 Applying Sledgehammer Bypasses..."
find src -type f -name "*.py" -exec sed -i 's/AttentionBackendName\.flash_2/AttentionBackendName\.torch/g' {} +
find src -type f -name "*.py" -exec sed -i 's/"flash_2"/"torch"/g' {} +
find src -type f -name "*.py" -exec sed -i "s/'flash_2'/'torch'/g" {} +
find src -type f -name "*.py" -exec sed -i 's/m\.apply_compile()/pass/g' {} +

echo "🎉 Environment is ready and bypassed!"
echo "⚠️  (If your terminal prompt does not say '(my_env)' right now, run: source /workspace/my_env/bin/activate)"