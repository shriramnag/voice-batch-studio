# Version 0.181.07-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"

# GPU को प्राथमिकता देना (Superfast Mode)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 सिस्टम {device} पर सुपरफास्ट मोड में चालू है...")

# AI मॉडल लोड करना
try:
    # हकलाना कम करने के लिए मॉडल को GPU मेमोरी में मजबूती से लोड करना
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"मॉडल लोड एरर: {e}")

def generate_voice(voice_sample, script, emotion, speed, language, remove_silence):
    if not voice_sample or not script:
        return None, "❌ गलती: सैंपल और स्क्रिप्ट डालें!"
    
    # हकलाना रोकने के लिए टेक्स्ट की गहरी सफ़ाई
    clean_text = script.replace("\n", " ").strip()
    output_path = f"vbs_fast_{int(time.time())}.wav"
    
    try:
        start_time = time.time()
        # AI आवाज़ जनरेशन - लय और गति में सुधार के साथ
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            # हकलाना कम करने के लिए ये सेटिंग्स जोड़ी गई हैं
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=5.0,
            enable_text_splitting=True
        )
        process_time = round(time.time() - start_time, 2)
        return output_path, f"✅ सफलता! {process_time}s में तैयार | शब्द: {len(script.split())}"
    except Exception as e:
        return None, f"❌ AI एरर: {str(e)}"

def count_words(text):
    return f"शब्दों की संख्या: {len(text.split())} / 10,000"

# आधुनिक डार्क और लाइट मोड इंटरफ़ेस
with gr.Blocks(theme='shivi/calm_cyan') as demo:
    gr.Markdown("# 🎙️ **वॉइस-बैच स्टूडियो प्रो v0.181.07**")
    gr.Markdown("*(Superfast GPU Mode Enabled)*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ **कंट्रोल पैनल**")
            voice_in = gr.Audio(label="अपना सैंपल दें", type="filepath")
            
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en", "es"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry"], value="Neutral", label="🎭 भावना")
            
            speed_sl = gr.Slider(0.7, 1.4, 1.0, step=0.05, label="⏩ आवाज़ की गति (Speed)")
            silence_btn = gr.Checkbox(label="🤫 फालतू सन्नाटा हटाएं", value=True)
            
            gen_btn = gr.Button("🚀 जेनरेट करें (Fast Mode)", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            word_display = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="अपनी कहानी यहाँ लिखें", lines=15, placeholder="लंबी स्क्रिप्ट यहाँ डालें...")
            
            script_in.change(count_words, inputs=[script_in], outputs=[word_display])
            
            gr.Markdown("### 🎧 **आउटपुट**")
            audio_out = gr.Audio(label="सुनें और डाउनलोड करें", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt, silence_btn], [audio_out, status])

demo.launch(share=True)
