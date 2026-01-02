import torch
import torch.nn.functional as F
from typing import List

def padding(wavs: List[torch.Tensor], size_unit: int) -> torch.Tensor:
    """
    सभी ऑडियो (wavs) को एक बराबर लंबाई में पैड करता है 
    ताकि वे size_unit के मल्टीपल (गुणांक) में हों।
    """
    lengths = [len(wav) for wav in wavs]
    max_len = max(lengths)
    
    # लंबाई को size_unit के अगले मल्टीपल तक बढ़ाना
    if max_len % size_unit != 0:
        max_len += (size_unit - (max_len % size_unit))
        
    padded_wavs = []
    for wav in wavs:
        pad_row = max_len - len(wav)
        # आखिरी में ज़ीरो जोड़कर लंबाई बराबर करना
        padded_wavs.append(F.pad(wav, (0, pad_row)))
        
    return torch.stack(padded_wavs)
