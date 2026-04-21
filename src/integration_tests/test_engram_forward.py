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

    # 3. Build the model directly on the device
    log.info("🏭 Assembling the factory...")
    model = config.build(init_device=device)
    model.init_weights()

    # 4. Create dummy data ON THE DEVICE
    log.info("📦 Prepping raw materials (tokens)...")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # 5. Forward Pass
    log.info("⚙️ Running the assembly line...")
    try:
        # Run the forward pass!
        output = model(input_ids=input_ids)
        
        log.info("\n🎉 SUCCESS! The Walking Skeleton is ALIVE! 🎉")
        log.info(f"Input shape:  {input_ids.shape}")
        log.info(f"Output shape: {output.shape} (Should be [Batch, SeqLen, VocabSize])")
        
    except Exception as e:
        log.error("\n💥 CRASH! The bulldozer hit a wall.")
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    run_engram_forward_test()