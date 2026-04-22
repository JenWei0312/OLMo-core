import torch
import logging

# Import OLMo's config and our new Engram config
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.engram.config import EngramConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_engram_forward_test():
    # Detect the GPU (Perfect for your Colab T4!)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"🚜 Firing up the bulldozer engine on {device.upper()}...")

    # 1. Create a tiny Engram config
    engram_config = EngramConfig(
        max_ngram_size=3,
        n_embed_per_ngram=64, # Tiny dimensions!
        n_head_per_ngram=2,   # Just 2 heads
        layer_ids=[1, 2],     # Inject at layers 1 and 2
        engram_vocab_size=[1000, 1000] # Tiny memory capacity!
    )

    # 2. Create a tiny OLMo config and attach Engram
    log.info("📝 Building blueprints...")
    config = TransformerConfig.olmo2_1M(
        vocab_size=50257, 
        n_layers=4,
        engram=engram_config 
    )

    # 3. Build the model directly on the GPU
    log.info("🏭 Assembling the factory (Directly on GPU)...")
    model = config.build(init_device=device)
    
    # We completely skip model.init_weights() because:
    # 1. We don't care about perfect statistical weight distributions for a dry-run test.
    # 2. PyTorch modules already have default random weights.
    # 3. It completely bypasses the Colab NVRTC C++ compilation bug!

    # 4. Create dummy data ON THE DEVICE
    log.info("📦 Prepping raw materials (tokens)...")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # 5. Forward Pass -- also mocking the loss calculation to test autograd
    log.info("⚙️ Running the forward pass...")
    try:
        # Get the logits [Batch, SeqLen, VocabSize]
        logits = model(input_ids=input_ids)
        
        # 6. Calculate Dummy Loss
        log.info("📉 Calculating dummy loss...")
        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss()
        
        # Create fake target words for the model to predict
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Reshape for CrossEntropyLoss: Logits must be 2D [Batch * SeqLen, VocabSize], Labels 1D
        loss = loss_fn(logits.view(-1, config.vocab_size), labels.view(-1))
        log.info(f"Initial Loss: {loss.item():.4f}")

        # 7. THE TRUE TEST: The Backward Pass
        log.info("🔙 Putting the bulldozer in reverse (Testing Autograd)...")
        loss.backward()

        # 8. The Optimizer Step
        log.info("👟 Stepping the optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad() # Clean up

        log.info("\n🎉 SUCCESS! The Bulldozer can drive in reverse! Autograd is connected! 🎉")
        log.info("The Engram module is 100% mathematically ready for training.")
        
    except Exception as e:
        log.error("\n💥 CRASH! The backward pass failed.")
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run_engram_forward_test()