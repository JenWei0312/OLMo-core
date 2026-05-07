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

# 3. Base Installation
echo "⚙️ Installing OLMo-core dependencies..."
pip install -e .[all]

# 4. The PyTorch Sync
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$CURRENT_TORCH" != *"2.6.0"* ]]; then
    echo "🔥 Syncing PyTorch 2.6.0..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
else
    echo "✅ PyTorch 2.6.0 is already installed."
fi

# 5. The Exorcism
echo "🧟 Exorcising Zombie Libraries..."
pip uninstall -y flash-attn cutlass torch_c_dlpack_ext

/*
# 6. The Phantom Bypasses (Flash Attention & Gantry)
echo "👻 Setting up Phantom Imports..."
# --- Flash Attention Fake ---
mkdir -p flash_attn/cute
touch flash_attn/__init__.py
echo "class Dummy: pass" > flash_attn/cute/__init__.py
echo "class Dummy: pass" > flash_attn/cute/interface.py

# --- Gantry Fake ---
mkdir -p gantry
touch gantry/__init__.py
echo "class Callback: pass" > gantry/callbacks.py
echo "class ExperimentFailedError(Exception): pass" > gantry/exceptions.py
cat << 'EOF' > gantry/api.py
class GitRepoState:
    @classmethod
    def from_env(cls):
        return cls()

class Recipe: pass
EOF
*/

# 6. PHANTOM IMPORTS (Upgraded with Decorator Support)
echo "👻 Setting up Phantom Imports..."
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
mkdir -p $SITE_PACKAGES/gantry

cat << 'EOF' > $SITE_PACKAGES/gantry/__init__.py
class Callback:
    @classmethod
    def register(cls, *args, **kwargs):
        # A dummy decorator that just returns the class unchanged
        return lambda x: x

class GantryCallback(Callback):
    pass
EOF
echo "✅ Phantom Beaker/Gantry module injected!"

# 7. The Sledgehammer Bypasses (PyTorch Core)
echo "🔨 Applying Sledgehammer Bypasses..."
find src -type f -name "*.py" -exec sed -i 's/AttentionBackendName\.flash_2/AttentionBackendName\.torch/g' {} +
find src -type f -name "*.py" -exec sed -i 's/"flash_2"/"torch"/g' {} +
find src -type f -name "*.py" -exec sed -i "s/'flash_2'/'torch'/g" {} +
find src -type f -name "*.py" -exec sed -i 's/m\.apply_compile()/pass/g' {} +

# 8. THE DATASET SYNC (FAKE DUMMY DATA)
echo "📊 Generating Fake Dummy Dataset..."
mkdir -p /workspace/dummy_data

# Use Python to instantly create an .npy file with 10 million random tokens
python -c "
import numpy as np
print('Generating random tokens...')
dummy_data = np.random.randint(0, 50257, size=(10000000,), dtype=np.uint16)
np.save('/workspace/dummy_data/000000.npy', dummy_data)
print('✅ Dummy dataset created!')
"

echo "🎉 Environment is completely built, bypassed, and data is loaded!"
echo "⚠️  (If your terminal prompt does not say '(my_env)' right now, run: source /workspace/my_env/bin/activate)"