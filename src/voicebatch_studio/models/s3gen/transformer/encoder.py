# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Encoder definition."""

from typing import Tuple, List, Optional, Union

import torch
from torch import nn

from voicebatch_studio.models.s3gen.transformer.encoder_layer import ConformerEncoderLayer
from voicebatch_studio.models.s3gen.transformer.utils import make_pad_mask


class ConformerEncoder(nn.Module):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        self_attention_layer_type: str = "selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        # यहाँ हम एम्बेडिंग और लेयर्स को डिफाइन करते हैं
        self.embed = None # Subsampling layer goes here
        self.after_norm = nn.LayerNorm(output_size, eps=1e-12)
        
        # Encoder Layers का निर्माण
        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                # Attention, FeedForward और Conv यहाँ पास किए जाते हैं
            ) for _ in range(num_blocks)
        ])

    def forward(
        self,
        xs_pad: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """पूरे ऑडियो डेटा को प्रोसेस करना।"""
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        
        for layer in self.encoders:
            xs_pad, masks, _, _ = layer(xs_pad, masks, None)
            
        xs_pad = self.after_norm(xs_pad)
        return xs_pad, masks
