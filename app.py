import os
import sys
import torch
import gradio as gr

# आपके 'src' फोल्डर को लिंक करना
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# सही फोल्डर 'voicebatch_studio' से इम्पोर्ट करना
from voicebatch_studio.tts_turbo import ChatterboxTurboTTS
from voicebatch_studio.mtl_tts import ChatterboxMultilingualTTS
from voicebatch_studio.vc import ChatterboxVC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- UI Functions ---
def run_turbo(text, reference_audio):
    # टर्बो मॉडल चलाने का लॉजिक
    return None 

def run_multilingual(text, lang, reference_audio):
    # बहुभाषी मॉडल चलाने का लॉजिक
    return None

# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ VoiceBatch Studio Pro")
    gr.Markdown("आपका अपना एआई वॉइस क्लोनिंग और प्रोसेसिंग सेंटर")

    with gr.Tabs():
        # टैब 1: टर्बो इंजन
        with gr.TabItem("🚀 Turbo TTS"):
            with gr.Row():
                with gr.Column():
                    t_text = gr.Textbox(label="टेक्स्ट लिखें", placeholder="यहाँ अपना संदेश डालें...")
                    t_ref = gr.Audio(label="रेफरेंस आवाज (Optional)", type="filepath")
                    t_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    t_out = gr.Audio(label="तैयार आवाज")

        # टैब 2: बहुभाषी (Multilingual)
        with gr.TabItem("🌍 Multilingual"):
            with gr.Row():
                with gr.Column():
                    m_text = gr.Textbox(label="टेक्स्ट")
                    m_lang = gr.Dropdown(choices=["hi", "en", "es", "fr"], label="भाषा चुनें", value="hi")
                    m_ref = gr.Audio(label="अपनी आवाज अपलोड करें", type="filepath")
                    m_btn = gr.Button("बनाएँ", variant="primary")
                with gr.Column():
                    m_out = gr.Audio(label="आउटपुट")

        # टैब 3: वॉइस कन्वर्जन
        with gr.TabItem("🎙️ Voice Conversion"):
            gr.Markdown("किसी भी आवाज को अपनी आवाज में बदलें")
            # VC का इंटरफेस यहाँ आएगा

demo.launch()
