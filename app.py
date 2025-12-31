# Version 0.181.15-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time
import shutil
from pydub import AudioSegment, effects

# नियमों को स्वीकार करना और परफॉरमेंस बढ़ाना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.set_num_threads(os.cpu_count())

# वॉयस मेमोरी सेटअप
VOICE_DIR = "custom_voices"
os.makedirs(VOICE_DIR, exist_ok=True)

def get_all_voices():
    # डिफ़ॉल्ट और अपलोड की गई आवाज़ें
    defaults = ["Joanne.wav", "Reginald voice.wav"]
    customs = [f for f in os.listdir(VOICE_DIR) if f.endswith('.wav')]
    return defaults + customs

print(f"🚀 टर्बो इंजन चालू: {device} | वर्शन: 0.181.15")

try:
    # मॉडल को तेज़ लोड करना
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"Error: {e}")

def save_new_voice(file):
    if file is None: return gr.update()
    filename = os.path.basename(file.name)
    dest = os.path.join(VOICE_DIR, filename)
    shutil.copy(file.name, dest)
    return gr.update(choices=get_all_voices(), value=filename)

def generate_voice(voice_name, script, emotion, speed, language, remove_silence, voice_enhance):
    if not voice_name or not script:
        return None, "❌ डेटा डालें!"
    
    # फाइल पाथ सेट करना
    voice_path = voice_name if os.path.exists(voice_name) else os.path.join(VOICE_DIR, voice_name)
    
    if not os.path.exists(voice_path):
        return None, f"❌ वॉयस फाइल '{voice_name}' नहीं मिली। कृपया उसे अपलोड करें।"

    output_path = f"vbs_mega_{int(time.time())}.wav"
    
    try:
        start_time = time.time()
        # लंबी स्क्रिप्ट को ऑटो-स्प्लिट करना (Long Script Fix)
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_path,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True # यह 80 शब्द वाली लिमिट खत्म कर देगा
        )
        
        # ऑडियो फिनिशिंग
        audio = AudioSegment.from_wav(output_path)
        if remove_silence: audio = effects.strip_silence(audio, silence_thresh=-45, padding=150)
        if voice_enhance: audio = effects.normalize(audio)
        audio.export(output_path, format="wav")
        
        duration = round(time.time() - start_time, 2)
        return output_path, f"✅ सुपरफास्ट जनरेशन पूरा! समय: {duration}s"
    except Exception as e:
        return None, f"❌ AI एरर: {str(e)}"

# --- UI (हरा प्रोग्रेस बार थीम) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Pro: Unlimited Edition**")
    gr.Markdown("### *Version 0.181.15-release (Super Speed Fix)*")
    
    with gr.Row():
        with gr.Column(scale=1):
            voice_select = gr.Dropdown(choices=get_all_voices(), label="वॉयस लाइब्रेरी", value="Joanne.wav")
            new_voice_upload = gr.File(label="नई आवाज़ जोड़ें (Save to Memory)", file_types=[".wav"])
            new_voice_upload.change(save_new_voice, inputs=[new_voice_upload], outputs=[voice_select])

            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited"], value="Neutral", label="🎭 इमोशन")
            
            speed_sl = gr.Slider(0.8, 1.3, 1.0, step=0.01, label="⏩ गति")
            silence_btn = gr.Checkbox(label="🤫 सन्नाटा हटाना", value=True)
            enhance_btn = gr.Checkbox(label="✨ आवाज़ निखारना", value=True)
            
            gen_btn = gr.Button("🚀 GENERATE (MEGA SPEED)", variant="primary")
            status = gr.Textbox(label="स्टेटस", interactive=False)

        with gr.Column(scale=2):
            word_counter = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="यहाँ लंबी स्क्रिप्ट पेस्ट करें (कोई लिमिट नहीं)", lines=18)
            script_in.change(lambda x: f"शब्दों की संख्या: {len(x.split())} / 10,000", inputs=[script_in], outputs=[word_counter])
            
            audio_out = gr.Audio(label="आउटपुट", type="filepath")

    gen_btn.click(generate_voice, [voice_select, script_in, emotion_opt, speed_sl, lang_opt, silence_btn, enhance_btn], [audio_out, status])

demo.launch(share=True)
