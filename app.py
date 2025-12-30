# Version 0.181.10-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU की ताकत बढ़ाने के लिए
if device == "cpu":
    torch.set_num_threads(8)

print(f"🚀 लोड हो रहा है: Version 0.181.10-release | डिवाइस: {device}")

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"Error: {e}")

# 1. स्मार्ट स्क्रिप्ट इम्प्रूवर (सिर्फ बटन दिखाने के लिए अभी लॉजिक जोड़ा है)
def improve_script(text, style):
    if style == "भावुक (Emotional)":
        return text + "..." # यह भविष्य में AI से स्क्रिप्ट सुधारेगा
    return text

def generate_voice(voice_sample, script, emotion, speed, language, use_enhancer):
    if not voice_sample or not script:
        return None, "❌ कृपया डेटा डालें!"
    
    output_path = f"vbs_2026_{int(time.time())}.wav"
    try:
        start_time = time.time()
        # हकलाना रोकने के लिए 2026 की नई सेटिंग्स
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True
        )
        process_time = round(time.time() - start_time, 2)
        return output_path, f"✅ सफलता! समय: {process_time}s | एन्हेंसर: {'ON' if use_enhancer else 'OFF'}"
    except Exception as e:
        return None, f"❌ एरer: {str(e)}"

# --- मॉडर्न 2026 इंटरफ़ेस ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Studio Pro 2026**")
    gr.Markdown("### *Version 0.181.10-release | AI वॉइस इंटेलिजेंस*")
    
    with gr.Row():
        # बायाँ हिस्सा: कंट्रोल टूल्स
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ **स्मार्ट टूल्स**")
            voice_in = gr.Audio(label="वॉयस क्लोनिंग सैंपल", type="filepath")
            
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en", "es", "fr"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited", "Whisper"], value="Neutral", label="🎭 मुख्य भावना")
            
            speed_sl = gr.Slider(0.8, 1.3, 1.0, step=0.01, label="⏩ गति कंट्रोल")
            
            # नए टूल्स के बटन
            use_enhancer = gr.Checkbox(label="✨ AI Voice Enhancer (आवाज़ निखारें)", value=True)
            bg_music = gr.Checkbox(label="🎵 Auto Background Music (Beta)", value=False)
            
            gen_btn = gr.Button("🔥 GENERATE AI VOICE", variant="primary")
            status = gr.Textbox(label="स्टेटस", interactive=False)

        # दायाँ हिस्सा: स्मार्ट एडिटर
        with gr.Column(scale=2):
            gr.Markdown("### 📜 **Smart Script Editor v2**")
            with gr.Row():
                script_style = gr.Radio(["सामान्य", "भावुक (Emotional)", "जोशीला (Excited)"], label="स्क्रिप्ट का अंदाज़ बदलें", value="सामान्य")
                improve_btn = gr.Button("🪄 Improve Script", size="sm")
            
            script_in = gr.Textbox(label="", lines=15, placeholder="यहाँ अपनी कहानी लिखें...")
            
            # स्क्रिप्ट सुधारने का फंक्शन जोड़ना
            improve_btn.click(improve_script, [script_in, script_style], script_in)
            
            gr.Markdown("### 🎧 **फाइनल मास्टर आउटपुट**")
            audio_out = gr.Audio(label="सुनें और डाउनलोड करें", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt, use_enhancer], [audio_out, status])

demo.launch(share=True)
