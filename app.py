# Version 0.181.16-release (High Processing Edition)
import os
import gradio as gr
from TTS.api import TTS
import torch
import time
import shutil
from pydub import AudioSegment, effects

# नियमों को स्वीकार करना और परफॉरमेंस को 100% पर सेट करना
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count()) # High Processing
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.set_num_threads(os.cpu_count())

# वॉयस मेमोरी और फोल्डर सेटअप
VOICE_DIR = "custom_voices"
os.makedirs(VOICE_DIR, exist_ok=True)

def get_all_voices():
    defaults = ["Joanne.wav", "Reginald voice.wav"]
    customs = [f for f in os.listdir(VOICE_DIR) if f.endswith('.wav')]
    return defaults + customs

print(f"🚀 हाई प्रोसेसिंग मोड: {device} | वर्शन: 0.181.16")

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"Model Load Error: {e}")

# --- स्मार्ट हुक लाइब्रेरी ---
def smart_hook_editor(text, category):
    if not text: return text
    hooks = {
        "सस्पेंस/डरावनी": "सावधान! जो आप सुनने वाले हैं, उसने हज़ारों लोगों की रातों की नींद उड़ा दी है... ",
        "फैक्ट्स/ज्ञान": "क्या आप जानते हैं? विज्ञान की दुनिया का एक ऐसा सच जो आज तक आपसे छुपाया गया... ",
        "कहानी/भावुक": "ज़िंदगी के मोड़ पर कभी-कभी ऐसी दास्ताँ सामने आती है, जो रूह को कंपा देती है... ",
        "मोटिवेशन/जोश": "वक्त आ गया है दुनिया को यह दिखाने का कि आप में कितनी आग बाकी है! "
    }
    return hooks.get(category, "") + text

def save_new_voice(file):
    if file is None: return gr.update()
    filename = os.path.basename(file.name)
    dest = os.path.join(VOICE_DIR, filename)
    shutil.copy(file.name, dest)
    return gr.update(choices=get_all_voices(), value=filename)

def generate_voice(voice_name, script, emotion, speed, language, remove_silence, voice_enhance):
    if not voice_name or not script:
        return None, "❌ डेटा डालें!"
    
    # वॉयस पाथ फिक्स (Error Fix)
    voice_path = voice_name if os.path.exists(voice_name) else os.path.join(VOICE_DIR, voice_name)
    if not os.path.exists(voice_path):
        return None, f"❌ वॉयस फाइल '{voice_name}' नहीं मिली।"

    output_path = f"vbs_high_res_{int(time.time())}.wav"
    
    try:
        start_time = time.time()
        # High Speed Generation
        tts.tts_to_file(
            text=script,
            speaker_wav=voice_path,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True
        )
        
        # पोस्ट प्रोसेसिंग (Enhancer & Silence)
        audio = AudioSegment.from_wav(output_path)
        if remove_silence: audio = effects.strip_silence(audio, silence_thresh=-45, padding=150)
        if voice_enhance: audio = effects.normalize(audio)
        audio.export(output_path, format="wav")
        
        duration = round(time.time() - start_time, 2)
        return output_path, f"✅ जनरेशन पूरा! समय: {duration}s"
    except Exception as e:
        return None, f"❌ AI Error Fix: {str(e)}"

# --- UI डिज़ाइन ---
custom_css = ".progress-bar { background-color: #28a745 !important; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=custom_css) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Pro: High-Processing 2026**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ **कंट्रोल पैनल**")
            voice_select = gr.Dropdown(choices=get_all_voices(), label="वॉयस लाइब्रेरी", value="Joanne.wav")
            new_voice_upload = gr.File(label="नई आवाज़ जोड़ें", file_types=[".wav"])
            new_voice_upload.change(save_new_voice, inputs=[new_voice_upload], outputs=[voice_select])

            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited"], value="Neutral", label="🎭 इमोशन")
            
            speed_sl = gr.Slider(0.7, 1.4, 1.0, step=0.01, label="⏩ स्पीड")
            silence_btn = gr.Checkbox(label="🤫 सन्नाटा हटाना", value=True)
            enhance_btn = gr.Checkbox(label="✨ आवाज़ निखारना", value=True)
            
            gen_btn = gr.Button("🚀 GENERATE (HIGH SPEED)", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🪄 **स्मार्ट हुक एडिटर v4**")
            with gr.Row():
                hook_cat = gr.Dropdown(["सस्पेंस/डरावनी", "फैक्ट्स/ज्ञान", "कहानी/भावुक", "मोटिवेशन/जोश"], label="हुक कैटेगरी")
                hook_btn = gr.Button("🪄 Add Hook")
            
            word_counter = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="यहाँ स्क्रिप्ट लिखें", lines=15)
            
            # लाइव अपडेट्स
            script_in.change(lambda x: f"शब्दों की संख्या: {len(x.split())} / 10,000", inputs=[script_in], outputs=[word_counter])
            hook_btn.click(smart_hook_editor, [script_in, hook_cat], script_in)
            
            audio_out = gr.Audio(label="आउटपुट", type="filepath")

    gen_btn.click(generate_voice, [voice_select, script_in, emotion_opt, speed_sl, lang_opt, silence_btn, enhance_btn], [audio_out, status])

demo.launch(share=True)
