import os
import gradio as gr
from TTS.api import TTS
import torch

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# AI मॉडल को लोड करना
print(f"AI मॉडल {device} पर लोड हो रहा है...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(voice_sample, script, emotion, speed, language):
    if not voice_sample or not script:
        return None, "❌ गलती: कृपया आवाज़ का नमूना और स्क्रिप्ट दोनों डालें!"
    
    output_path = "vbs_2025_final.wav"
    
    try:
        # असली इंसानी आवाज़ और भावनाओं के साथ ऑडियो बनाना
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed
        )
        return output_path, f"✅ सफलता: ऑडियो तैयार है! कुल शब्द: {len(script.split())}"
    except Exception as e:
        return None, f"❌ AI एरर: {str(e)}"

# शब्दों को गिनने वाला फंक्शन
def count_words(text):
    words = len(text.split())
    return f"शब्दों की संख्या: {words} / 10,000"

# आधुनिक 2025 डिज़ाइन (Dark Theme)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")) as demo:
    gr.Markdown("# 🎙️ **वॉइस-बैच स्टूडियो प्रो v5.0**")
    gr.Markdown("### *इंसानी भावनाओं के साथ AI आवाज़ क्लोनिंग*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ **सेटिंग्स**")
            voice_in = gr.Audio(label="अपनी आवाज़ का सैंपल यहाँ डालें", type="filepath")
            
            with gr.Row():
                lang_opt = gr.Dropdown(
                    choices=["hi", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko"],
                    value="hi", label="🌍 भाषा (Language)"
                )
                emotion_opt = gr.Dropdown(
                    choices=["Neutral", "Sad", "Angry", "Happy", "Surprise", "Whispering"], 
                    value="Neutral", label="🎭 भावना (Emotion)"
                )
            
            speed_sl = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="⏩ आवाज़ की गति (Speed)")
            gen_btn = gr.Button("🔥 आवाज़ जेनरेट करें", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📜 **स्क्रिप्ट (अधिकतम 10,000 शब्द)**")
            word_count_display = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(
                label="अपनी कहानी यहाँ पेस्ट करें", 
                lines=15, 
                placeholder="यहाँ लिखना शुरू करें..."
            )
            
            # स्क्रिप्ट लिखते समय शब्दों की गिनती अपडेट होगी
            script_in.change(count_words, inputs=[script_in], outputs=[word_count_display])
            
            gr.Markdown("### 🎧 **आउटपुट (सुनें और डाउनलोड करें)**")
            audio_out = gr.Audio(label="तैयार ऑडियो", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt], [audio_out, status])

# ऐप को शेयर लिंक के साथ चालू करना
demo.launch(share=True)
