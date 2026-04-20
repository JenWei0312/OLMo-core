import logging
import math
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from fnmatch import fnmatch
from itertools import cycle, islice
from typing import TYPE_CHECKING, Dict, List, Optional, cast

from olmo_core.config import UNSET, DType, StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.utils import ensure_multiple_of

@beta_feature
@dataclass
class EngramConfig:
    """Configuration for the Engram sparse memory module."""
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    kernel_size: int = 4
    
    # The layers where Engram will be injected (0-indexed)
    layer_ids: List[int] = field(default_factory=lambda: [1, 14]) # layer 2 and 15
    
    # Capacity: [2-gram capacity, 3-gram capacity]
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    
    pad_id: int = 2
    seed: int = 0