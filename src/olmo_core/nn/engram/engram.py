import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

from .config import EngramConfig # Import the config you just made!

# ==========================================
# 1. The Offline Data Hack (Tokenizer)
# ==========================================
class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
            normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL), normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new, key2new, new_tokens = {}, {}, []
        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)  

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Convert the numpy lookup array into a PyTorch tensor lazily
        if not isinstance(self.lookup_table, torch.Tensor):
            self.lookup_table = torch.tensor(self.lookup_table, dtype=torch.long)
        
        # Ensure it's on the exact same GPU as the incoming batch
        self.lookup_table = self.lookup_table.to(input_ids.device)
        
        pos_mask = input_ids >= 0
        out = input_ids.clone()
        # Apply the mapping directly on the GPU
        out[pos_mask] = self.lookup_table[input_ids[pos_mask]]
        return out

# ==========================================
# 2. The 3D Short Convolution (De-Hyper-Connected)
# ==========================================
class ShortConv(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        # De-hyper-connected: We just use d_model directly!
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model, # Depthwise convolution
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm = nn.RMSNorm(d_model, eps=norm_eps) 
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, L, D) -> Standard OLMo hidden states!
        Output: (B, L, D)
        """
        B, T, C = x.shape
        x_norm = self.norm(x)
        
        # Conv1d expects (Batch, Channels, Length)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T] # Truncate padding
        y_bct = self.act_fn(y_bct)
        
        # Transpose back to (Batch, Length, Channels)
        y = y_bct.transpose(1, 2).contiguous()
        return y
    
# ==========================================
# 3. Hash Mapping & Embeddings
# ==========================================
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

# 1. Inherit from nn.Module
class NgramHashMapping(nn.Module):
    def __init__(self, config: EngramConfig):
        super().__init__() # Must call this first!
        self.vocab_size_per_ngram = config.engram_vocab_size
        self.max_ngram_size = config.max_ngram_size
        self.n_head_per_ngram = config.n_head_per_ngram
        self.pad_id = config.pad_id
        self.layer_ids = config.layer_ids

        self.compressed_tokenizer = CompressedTokenizer()            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        M_max = int(np.iinfo(np.int64).max // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        # We calculate the pure Python/NumPy math first...
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(config.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

        # ---------------------------------------------------------
        # 2. THE FIX: Register the static data as PyTorch Buffers
        # By dynamically naming them based on layer_id, PyTorch will 
        # permanently pin these to the correct GPU at initialization.
        # ---------------------------------------------------------
        for layer_id in self.layer_ids:
            # Register Multipliers
            mults_tensor = torch.tensor(self.layer_multipliers[layer_id], dtype=torch.long)
            self.register_buffer(f"multipliers_layer_{layer_id}", mults_tensor)
            
            # Register Head Vocab Sizes (Convert list of lists to a 2D tensor)
            sizes_tensor = torch.tensor(self.vocab_size_across_layers[layer_id], dtype=torch.long)
            self.register_buffer(f"head_sizes_layer_{layer_id}", sizes_tensor)

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                current_prime_search_start = vocab_size - 1
                for _ in range(self.n_head_per_ngram):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers
    
    def _get_ngram_hashes(self, input_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        import torch.nn.functional as F
        
        x = input_ids # Now inherently a PyTorch tensor!
        B, T = x.shape
        
        # 3. THE FIX: Retrieve the pinned GPU buffers using getattr()
        # No more .to(device) calls or torch.tensor() instantiations!
        multipliers = getattr(self, f"multipliers_layer_{layer_id}")
        head_vocab_sizes = getattr(self, f"head_sizes_layer_{layer_id}")



        def shift_k(k: int) -> torch.Tensor:
            if k == 0: return x
            # F.pad pads from the last dimension backwards: (pad_left, pad_right)
            padded = F.pad(x, (k, 0), value=self.pad_id)
            return padded[:, :T]

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            # force mix to be positive tp avoid deviding by negative number
            mix = torch.abs(mix)
            
            # Take the 1D array of `head_vocab_sizes`` based on `n_gram_index`
            current_head_sizes = head_vocab_sizes[n_gram_index]
            
            # Unsqueeze adds the n_heads dimension, then modulo broadcasts
            hashes = mix.unsqueeze(-1) % current_head_sizes  # (B, T, n_heads)
            all_hashes.append(hashes)

        # Concatenate along the last axis to make it (B, T, 8)
        return torch.cat(all_hashes, dim=2)

    def hash(self, input_ids: torch.Tensor):
        # Everything stays as tensors
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers
    

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N, D):
        super().__init__()
        self.list_of_N = list_of_N
        self.D = D
        
        # Calculate offsets safely
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
            
        # Register as a buffer so it lives on the GPU
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        # Total size of the massive embedding table
        self.total_size = sum(list_of_N)
        self.embedding = nn.Embedding(self.total_size, D)

    def forward(self, input_ids: torch.Tensor):
        # 1. Force precise broadcasting shape (1, 1, Heads) for Triton
        offsets = self.offsets.view(1, 1, -1)
        indices = input_ids + offsets
        
        # 2. 🛑 THE IRONCLAD SAFETY NET 🛑
        # Force Triton's C++ kernel to respect the physical memory bounds.
        # This completely intercepts any compiler hallucinations or XOR overflows.
        max_valid_index = self.total_size - 1
        safe_indices = torch.clamp(indices, min=0, max=max_valid_index)
        
        return self.embedding(safe_indices)
    
# ==========================================
# 4. The Main Engram Block (De-Hyper-Connected)
# ==========================================
class Engram(nn.Module):
    def __init__(self, config: EngramConfig, layer_id: int, d_model: int):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = d_model
        
        # 1. Setup Hash Mapping
        self.hash_mapping = NgramHashMapping(config=config)
        
        # 2. Setup Embedding Table
        list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        D = config.n_embed_per_ngram // config.n_head_per_ngram
        self.multi_head_embedding = MultiHeadEmbedding(list_of_N=list_of_N, D=D)
        
        # 3. Setup Short Conv (3D instead of 4D)
        self.short_conv = ShortConv(
            d_model=d_model,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
        )
        
        # 4. Gating Projections (No lists of norms needed anymore!)
        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, d_model)
        self.key_proj = nn.Linear(engram_hidden_size, d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, L, D]
        input_ids: [B, L]
        """
        # 1. Pure GPU Hashing - NO CPU FALLBACK!
        hash_dict = self.hash_mapping.hash(input_ids)
        hash_input_ids = hash_dict[self.layer_id]
        
        # 2. Retrieve Memory
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        
        # 3. Context-Aware Gating (3D Math)
        key = self.norm1(self.key_proj(embeddings))
        query = self.norm2(hidden_states)
        
        gate = (key * query).sum(dim=-1) / math.sqrt(self.d_model)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)
        
        # 4. Output Injection
        value = gate * self.value_proj(embeddings)
        output = value + self.short_conv(value)
        return output