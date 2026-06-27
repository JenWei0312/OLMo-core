import torch
import torch.nn as nn
import logging
import traceback # <--- WE WILL NEVER FLY BLIND AGAIN!

# Import OLMo's configs
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.moe.moe import MoEConfig, MoERouterConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.engram.config import EngramConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def test_config(name: str, config: TransformerConfig, device: str, input_ids: torch.Tensor, labels: torch.Tensor):
    log.info(f"\n{'='*50}")
    log.info(f"🧪 TESTING: {name}")
    log.info(f"{'='*50}")
    
    try:
        log.info("🏭 Assembling the factory...")
        model = config.build(init_device=device)
        
        log.info("⚙️ Running forward pass...")
        logits = model(input_ids=input_ids)
        
        log.info("📉 Calculating loss...")
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, config.vocab_size), labels.view(-1))
        
        log.info("🔙 Testing backward pass (Autograd)...")
        loss.backward()
        
        log.info("👟 Stepping optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()
        
        log.info(f"✅ SUCCESS! {name} is fully compatible with Engram.")
        return True
    except Exception as e:
        log.error(f"❌ CRASH on {name}!")
        log.error(f"Error: {e}")
        # Print the exact line and kernel that failed!
        traceback.print_exc() 
        return False

def run_2x2_grid_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"🚜 Firing up the 2x2 Ablation Grid on {device.upper()}...\n")

    # 1. Global Engram Config
    engram_config = EngramConfig(
        max_ngram_size=3, n_embed_per_ngram=128, n_head_per_ngram=4,   
        layer_ids=[1, 2], engram_vocab_size=[1000, 1000] 
    )

    # 2. Setup dummy data once
    batch_size, seq_len, vocab_size = 2, 16, 50257
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    experiments = {}

    # ---------------------------------------------------------
    # Cell 1: Attention + Dense FFN (Baseline)
    # ---------------------------------------------------------
    cfg_attn_dense = TransformerConfig.olmo2_1M(vocab_size=vocab_size, n_layers=4, engram=engram_config)
    cfg_attn_dense.d_model = 128 # Override microscopic dimensions!
    cfg_attn_dense.block.sequence_mixer.n_heads = 4 #<- it's in the sequence_mixer config
    experiments["1. Attention + Dense FFN (Baseline)"] = cfg_attn_dense

    # ---------------------------------------------------------
    # Cell 2: Attention + MoE
    # ---------------------------------------------------------
    cfg_attn_moe = TransformerConfig.olmo2_1M(vocab_size=vocab_size, n_layers=4, engram=engram_config)
    cfg_attn_moe.d_model = 128
    cfg_attn_moe.block.sequence_mixer.n_heads = 4   #<- it's in the sequence_mixer config
    
    # Mutate the object DIRECTLY (No dicts!)
    cfg_attn_moe.block.name = "moe"
    cfg_attn_moe.block.feed_forward = None  # Turn off the dense FFN
    cfg_attn_moe.block.feed_forward_moe = MoEConfig(num_experts=8, router=MoERouterConfig(top_k=2))
    
    experiments["2. Attention + MoE"] = cfg_attn_moe

    # ---------------------------------------------------------
    # Cell 3: GDN (Linear RNN) + Dense FFN
    # ---------------------------------------------------------
    cfg_gdn_dense = TransformerConfig.olmo2_1M(vocab_size=vocab_size, n_layers=4, engram=engram_config)
    cfg_gdn_dense.d_model = 128
    # Explicitly tell GDN to use 4 heads so it doesn't crash computing group sizes
    cfg_gdn_dense.block.sequence_mixer = GatedDeltaNetConfig()
    cfg_gdn_dense.block.sequence_mixer.n_heads=4    # doing the same thing for the GDN config
    experiments["3. Linear RNN (GDN) + Dense FFN"] = cfg_gdn_dense

    # ---------------------------------------------------------
    # Cell 4: GDN (Linear RNN) + MoE
    # ---------------------------------------------------------
    cfg_gdn_moe = TransformerConfig.olmo2_1M(vocab_size=vocab_size, n_layers=4, engram=engram_config)
    cfg_gdn_moe.d_model = 128
    
    # Your brilliant GDN hack
    cfg_gdn_moe.block.sequence_mixer = GatedDeltaNetConfig()
    cfg_gdn_moe.block.sequence_mixer.n_heads = 4
    
    # Mutate the object DIRECTLY (No dicts!)
    cfg_gdn_moe.block.name = "moe"
    cfg_gdn_moe.block.feed_forward = None  # Turn off the dense FFN
    cfg_gdn_moe.block.feed_forward_moe = MoEConfig(num_experts=8, router=MoERouterConfig(top_k=2))
    
    experiments["4. Linear RNN (GDN) + MoE"] = cfg_gdn_moe

    # ---------------------------------------------------------

    # 4. Execute the Grid
    results = {}
    for name, config in experiments.items():
        results[name] = test_config(name, config, device, input_ids, labels)

    # 5. Final Report
    log.info(f"\n{'='*50}")
    log.info("📊 FINAL 2x2 GRID RESULTS:")
    log.info(f"{'='*50}")
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        log.info(f"{status} | {name}")

if __name__ == "__main__":
    run_2x2_grid_test()