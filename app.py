
# Version 0.181.01-release
import os
import gradio as gr
from TTS.api import TTS
import torch

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# AI मॉडल लोड करना - यहाँ हमने एरर को रोकने के लिए बदलाव किया है
print(f"AI मॉडल {device} पर लोड हो रहा है...")
try:
    # बिना किसी एक्स्ट्रा कोडेक के लोड करने की कोशिश
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"मॉडल लोड करने में दिक्कत: {e}")

def generate_voice(voice_sample, script, emotion, speed, language):
    if not voice_sample or not script:
        return None, "❌ गलती: आवाज़ का नमूना और स्क्रिप्ट डालें!"
    
    output_path = "vbs_final_output.wav"
    
    try:
        # असली इंसानी आवाज़ और भावनाओं के साथ जनरेशन
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed
        )
        return output_path, f"✅ सफलता! शब्दों की गिनती: {len(script.split())}"
    except Exception as e:
        return None, f"❌ AI एरर: {str(e)}"

# शब्दों को गिनने वाला फंक्शन
def count_words(text):
    words = len(text.split())
    return f"शब्दों की संख्या: {words} / 10,000"

# आधुनिक 2025 डार्क डिज़ाइन
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")) as demo:
    gr.Markdown("# 🎙️ **वॉइस-बैच स्टूडियो प्रो**")
    gr.Markdown("### *Version 0.181.01-release*")
    
    with gr.Row():
        with gr.Column(scale=1):
            voice_in = gr.Audio(label="आवाज़ का नमूना (Voice Sample)", type="filepath")
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en", "es", "fr"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Angry", "Happy"], value="Neutral", label="🎭 भावना")
            speed_sl = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="⏩ गति")
            gen_btn = gr.Button("🔥 आवाज़ जेनरेट करें", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            word_count_display = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
            script_in.change(count_words, inputs=[script_in], outputs=[word_count_display])
            audio_out = gr.Audio(label="यहाँ से सुनें और डाउनलोड करें", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt], [audio_out, status])

demo.launch(share=True)
