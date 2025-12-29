# Version 0.181.05-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 सिस्टम {device} पर सुपरफास्ट मोड में चालू है...")

# AI मॉडल लोड करना
try:
    # स्पीड बढ़ाने के लिए मॉडल को GPU पर प्राथमिकता दी गई है
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"मॉडल लोड एरर: {e}")

def generate_voice(voice_sample, script, emotion, speed, language):
    if not voice_sample or not script:
        return None, "❌ गलती: आवाज़ का नमूना और स्क्रिप्ट दोनों डालें!"
    
    # टेक्स्ट की सफ़ाई ताकि दूसरी भाषा न आए
    clean_script = script.replace("\n", " ").strip()
    output_path = f"vbs_final_{int(time.time())}.wav"
    
    try:
        start_time = time.time()
        # AI आवाज़ जेनरेट करना (बेहतर लय के साथ)
        tts.tts_to_file(
            text=clean_script,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True # लंबे वाक्यों को बिना हकलाए बोलने के लिए
        )
        end_time = time.time()
        process_speed = round(end_time - start_time, 2)
        
        return output_path, f"✅ सफलता! शब्द: {len(script.split())} | समय: {process_speed}s"
    except Exception as e:
        return None, f"❌ AI एरर: {str(e)}"

def count_words(text):
    words = len(text.split())
    return f"शब्दों की संख्या: {words} / 10,000"

# आधुनिक डार्क डिज़ाइन
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")) as demo:
    gr.Markdown("# 🎙️ **वॉइस-बैच स्टूडियो प्रो**")
    gr.Markdown("### *Version 0.181.05-release (Superfast & Realistic)*")
    
    with gr.Row():
        with gr.Column(scale=1):
            voice_in = gr.Audio(label="अपना स्पष्ट वॉयस सैंपल दें", type="filepath")
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry"], value="Neutral", label="🎭 भावना")
            speed_sl = gr.Slider(0.8, 1.5, 1.0, step=0.05, label="⏩ गति (Speed)")
            gen_btn = gr.Button("🚀 आवाज़ जेनरेट करें (Fast)", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            word_count_display = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="अपनी कहानी यहाँ लिखें", lines=15)
            script_in.change(count_words, inputs=[script_in], outputs=[word_count_display])
            audio_out = gr.Audio(label="तैयार ऑडियो (Clear Voice)", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt], [audio_out, status])

demo.launch(share=True)
