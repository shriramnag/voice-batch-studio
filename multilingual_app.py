import os
import sys
import random
import numpy as np
import torch
import gradio as gr

# --- 1. PATH FIX: यह आपके src फोल्डर को पायथन से जोड़ता है ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# --- 2. IMPORT FIX: 'chatterbox' की जगह 'voicebatch_studio' ---
from voicebatch_studio.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 VoiceBatch Studio running on device: {DEVICE}")

# बाकी का पूरा कोड (LANGUAGE_CONFIG, UI Helpers, आदि) वही रहेगा जो आपने ऊपर दिया है...
# बस सुनिश्चित करें कि नीचे दिए गए फंक्शन्स में भी 'MODEL' लोड करने का तरीका सही हो।

MODEL = None

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            # यहाँ भी सुनिश्चित करें कि यह आपके क्लास से लोड हो रहा है
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

# ... (यहाँ से आपका बाकी का कोड शुरू होता है जो आपने भेजा था) ...
