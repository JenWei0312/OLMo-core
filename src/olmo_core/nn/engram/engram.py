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
    
    def __call__(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        out[pos_mask] = self.lookup_table[arr[pos_mask]]
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

class NgramHashMapping:
    def __init__(self, config: EngramConfig):
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
        
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(config.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

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

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            return np.pad(x, ((0, 0), (k, 0)), mode='constant', constant_values=self.pad_id)[:, :T]

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                all_hashes.append((mix % mod).astype(np.int64, copy=False))
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(num_embeddings=sum(list_of_N), embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids + self.offsets)
    
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
        # 1. CPU Hashing -> GPU Tensor mapping (CRITICAL FIX)
        numpy_hashes = self.hash_mapping.hash(input_ids.cpu().numpy())[self.layer_id]
        hash_input_ids = torch.from_numpy(numpy_hashes).to(hidden_states.device)
        
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