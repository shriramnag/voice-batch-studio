# Version 0.181.02-release
import os
import gradio as gr
from TTS.api import TTS
import torch
from pydub import AudioSegment, effects

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# AI मॉडल लोड करना
print(f"AI मॉडल {device} पर चालू है...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_silence(audio_path):
    # सन्नाटा हटाने का फंक्शन
    audio = AudioSegment.from_wav(audio_path)
    audio = effects.strip_silence(audio, silence_thresh=-40)
    audio.export(audio_path, format="wav")
    return audio_path

def generate_voice(voice_sample, script, emotion, speed, language, remove_silence):
    if not voice_sample or not script:
        return None, "❌ गलती: सैंपल और स्क्रिप्ट डालें!"
    
    output_path = "vbs_final_output.wav"
    try:
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed
        )
        
        if remove_silence:
            output_path = clean_silence(output_path)
            
        return output_path, "✅ ऑडियो तैयार है!"
    except Exception as e:
        return None, f"❌ एरर: {str(e)}"

# आधुनिक इंटरफ़ेस
with gr.Blocks(theme=gr.themes.Default()) as demo:
    # डार्क/लाइट मोड का बटन अपने आप Gradio में ऊपर आता है
    gr.Markdown("# 🎙️ **वॉइस-बैच स्टूडियो प्रो v0.181.02**")
    
    with gr.Row():
        with gr.Column():
            voice_in = gr.Audio(label="आवाज़ का नमूना", type="filepath")
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en", "es"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry"], value="Neutral", label="🎭 भावना")
            
            silence_btn = gr.Checkbox(label="🤫 सन्नाटा हटाएं (Silence Remover)", value=True)
            speed_sl = gr.Slider(0.5, 2.0, 1.0, label="⏩ गति")
            gen_btn = gr.Button("🔥 आवाज़ जेनरेट करें", variant="primary")

        with gr.Column():
            script_in = gr.Textbox(label="अपनी स्क्रिप्ट (10,000 शब्द)", lines=12)
            audio_out = gr.Audio(label="सुनें और डाउनलोड करें", type="filepath")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt, silence_btn], [audio_out, status])

demo.launch(share=True)
