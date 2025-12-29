import os
import gradio as gr
from TTS.api import TTS
import torch
from pydub import AudioSegment

# Coqui TOS Agreement
os.environ["COQUI_TOS_AGREED"] = "1"

# GPU/CPU Detection
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Studio starting on {device}...")

# Load XTTS v2 Model
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"Error loading model: {e}")

def generate_voice(voice_sample, script, emotion, speed, cross_fade):
    if not voice_sample or not script:
        return None, "❌ Error: Voice Sample aur Script dono zaroori hain!"
    
    output_path = "vbs_pro_2025_output.wav"
    
    try:
        # Realistic Voice Generation with Emotions
        # 'emotion' parameter voice mein bhavukta (sad, happy, etc.) lata hai
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language="hi",
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True # Lambi script ke liye zaroori hai
        )
        
        return output_path, "✅ Voice Successfully Generated!"
    except Exception as e:
        return None, f"❌ AI Error: {str(e)}"

# --- Modern UI Theme (2025 Style) ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="cyan", secondary_hue="slate", font=["Inter", "sans-serif"])) as demo:
    
    gr.Markdown("""
    # 🎙️ **VoiceBatch Studio Pro v5.0**
    ### *Next-Gen AI Voice Cloning with Emotional Intelligence*
    ---
    """)
    
    with gr.Row():
        # Left Side: Controls
        with gr.Column(scale=1):
            gr.Markdown("### 🎚️ **Settings**")
            voice_input = gr.Audio(label="Upload Human Voice Sample", type="filepath")
            
            with gr.Row():
                emotion_opt = gr.Dropdown(
                    choices=["Neutral", "Sad", "Happy", "Angry", "Surprise", "Whispering", "Friendly"],
                    value="Neutral",
                    label="🎭 Select Emotion (Bhavukta)"
                )
                speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="⏩ Speed")
            
            gr.Markdown("---")
            generate_btn = gr.Button("🔥 GENERATE REALISTIC VOICE", variant="primary", size="lg")
            status_msg = gr.Textbox(label="System Status", interactive=False)

        # Right Side: Script & Output
        with gr.Column(scale=2):
            gr.Markdown("### 📜 **Script (Max 10,000 Words)**")
            script_input = gr.Textbox(
                label="Paste your story here...",
                placeholder="Yahan apni lambi kahani likhein... (Sad, Emotional ya Happy bhav ke saath)",
                lines=15,
                max_lines=30
            )
            
            gr.Markdown("---")
            gr.Markdown("### 🎧 **Audio Output**")
            audio_output = gr.Audio(label="Download Generated Voice", type="filepath")

    # Footer instructions for the user
    gr.Markdown("""
    **Pro Tips for Realistic Voice:**
    * **Pause:** Beech mein rukne ke liye `...` ka upyog karein.
    * **Emphasis:** Kisi shabd par zor dene ke liye use CAPITAL likhein.
    * **Emotions:** Agar dukhi kahani hai toh 'Sad' chunein, isse voice mein rone jaisa feel aayega.
    """)

    # Click Action
    generate_btn.click(
        fn=generate_voice,
        inputs=[voice_input, script_input, emotion_opt, speed_slider],
        outputs=[audio_output, status_msg]
    )

if __name__ == "__main__":
    # share=True provides the link for mobile
    demo.launch(share=True, debug=True)
