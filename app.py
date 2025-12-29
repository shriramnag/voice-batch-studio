import os
import gradio as gr
from TTS.api import TTS
import torch

# Coqui TOS Agreement
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# AI Model Load karna
print(f"AI Model loading on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(voice_sample, script, emotion, speed):
    if not voice_sample or not script:
        return None, "❌ Error: Script aur Voice Sample dono zaroori hain!"
    
    output_path = "vbs_2025_final.wav"
    
    try:
        # Realistic Voice with Emotions
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language="hi",
            file_path=output_path,
            emotion=emotion,
            speed=speed
        )
        return output_path, "✅ Voice Successfully Generated! Neeche se download karein."
    except Exception as e:
        return None, f"❌ AI Error: {str(e)}"

# Modern 2025 UI Design
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Studio Pro v5.0**")
    gr.Markdown("### *Next-Gen Emotional Voice Cloning*")
    
    with gr.Row():
        with gr.Column():
            voice_in = gr.Audio(label="Apna Voice Sample Upload Karein", type="filepath")
            with gr.Row():
                emotion_opt = gr.Dropdown(
                    choices=["Neutral", "Sad", "Angry", "Happy", "Surprise", "Whispering"], 
                    value="Neutral", label="🎭 Emotion Chunein"
                )
                speed_sl = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="⏩ Speed")
            gen_btn = gr.Button("🔥 GENERATE VOICE", variant="primary")
            status = gr.Textbox(label="System Status")

        with gr.Column():
            script_in = gr.Textbox(label="Apni Script Yahan Likhein (10,000 words)", lines=12)
            audio_out = gr.Audio(label="Download Audio Here", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl], [audio_out, status])

demo.launch(share=True)
