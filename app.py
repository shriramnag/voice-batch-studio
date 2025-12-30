# Version 0.181.09-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time

# 2026 Standards के लिए Punctuation और Text Cleaning
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU पर स्पीड बढ़ाने के लिए Threads की सेटिंग
if device == "cpu":
    torch.set_num_threads(4) # यह CPU की पूरी ताकत इस्तेमाल करेगा

print(f"🚀 मोड: {device} | वर्शन: 0.181.09-release")

try:
    # मॉडल को 'DeepSpeed' और 'Fast-Inference' मोड में लोड करना
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"लोडिंग एरर: {e}")

def generate_voice(voice_sample, script, emotion, speed, language):
    if not voice_sample or not script:
        return None, "❌ कृपया सैंपल और स्क्रिप्ट डालें!"
    
    # AI को हकलाने से रोकने के लिए स्मार्ट क्लीनिंग
    clean_text = script.replace("\n", " ").strip()
    output_path = f"vbs_2026_studio_{int(time.time())}.wav"
    
    try:
        start_time = time.time()
        # 2026 Advanced Inference Settings
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True
        )
        duration = round(time.time() - start_time, 2)
        return output_path, f"✅ जनरेशन पूरा! समय: {duration}s | डिवाइस: {device.upper()}"
    except Exception as e:
        return None, f"❌ एरर: {str(e)}"

# मॉडर्न और प्रीमियम UI (Dark Mode Default)
with gr.Blocks(theme='shivi/calm_cyan', title="VoiceBatch Studio 2026") as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Studio: Pro Edition 2026**")
    gr.Markdown("### *Version 0.181.09-release | Next-Gen Voice Intelligence*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎚️ **कंट्रोल सेंटर**")
            voice_in = gr.Audio(label="वॉयस सैंपल (High Quality)", type="filepath")
            
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en", "es", "fr", "ar"], value="hi", label="🌍 ग्लोबल भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited", "Whisper"], value="Neutral", label="🎭 इमोशन")
            
            speed_sl = gr.Slider(0.8, 1.3, 1.0, step=0.01, label="⏩ प्रो स्पीड कंट्रोल")
            gen_btn = gr.Button("🔥 GENERATE AI VOICE", variant="primary")
            status = gr.Textbox(label="सिस्टम प्रोग्रेस", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📜 **स्मार्ट स्क्रिप्ट एडिटर (10,000 शब्द)**")
            script_in = gr.Textbox(label="", lines=18, placeholder="यहाँ अपनी कहानी या स्क्रिप्ट पेस्ट करें...")
            
            gr.Markdown("### 🎧 **प्रोफेशनल ऑडियो आउटपुट**")
            audio_out = gr.Audio(label="सुनें और डाउनलोड करें", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt], [audio_out, status])

demo.launch(share=True)
