# VoiceBatch Studio Pro - Utilities for Class Mapping
# Based on CosyVoice Architecture

import torch

# Internal imports from our transformer module
from ..transformer.activation import Swish
from ..transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    LegacyLinearNoSubsampling
)
from ..transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    WhisperPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
    EspnetRelPositionalEncoding
)
from ..transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention
)

# Activation mapping for our models
VOICEBATCH_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

# Subsampling mapping
VOICEBATCH_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity
}

# Embedding/Positional Encoding mapping
VOICEBATCH_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

# Attention mechanism mapping
VOICEBATCH_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
