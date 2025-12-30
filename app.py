# Version 0.181.13-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time
from pydub import AudioSegment, effects

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU थ्रेड्स को मैक्सिमम करना (स्पीड के लिए)
if device == "cpu":
    torch.set_num_threads(os.cpu_count())

# डिफॉल्ट आवाजों के लिए फोल्डर बनाना
os.makedirs("default_voices", exist_ok=True)

print(f"🚀 टर्बो मोड चालू: {device} | वर्शन: 0.181.13-release")

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"मॉडल लोड एरर: {e}")

# --- एडवांस हुक लाइब्रेरी ---
def smart_hook_editor(text, category):
    if not text: return text
    
    hooks = {
        "सस्पेंस/डरावनी": "सावधान! जो आप सुनने वाले हैं, उसने हज़ारों लोगों की रातों की नींद उड़ा दी है... ",
        "फैक्ट्स/ज्ञान": "क्या आप जानते हैं? विज्ञान की दुनिया का एक ऐसा सच जो आज तक आपसे छुपाया गया... ",
        "कहानी/भावुक": "ज़िंदगी के मोड़ पर कभी-कभी ऐसी दास्ताँ सामने आती है, जो रूह को कंपा देती है... ",
        "मोटिवेशन/जोश": "वक्त आ गया है दुनिया को यह दिखाने का कि आप में कितनी आग बाकी है! "
    }
    
    selected_hook = hooks.get(category, "")
    return selected_hook + text

# --- ऑडियो एनहांसर और सन्नाटा हटाने वाला ---
def finalize_audio(file_path, remove_silence, enhance):
    audio = AudioSegment.from_wav(file_path)
    if remove_silence:
        audio = effects.strip_silence(audio, silence_thresh=-45, padding=150)
    if enhance:
        audio = effects.normalize(audio)
    audio.export(file_path, format="wav")
    return file_path

def generate_voice(voice_sample, script, emotion, speed, language, remove_silence, voice_enhance):
    if not voice_sample or not script:
        return None, "❌ डेटा डालें!"
    
    clean_text = script.replace("\n", " ").strip()
    output_path = f"vbs_13_final_{int(time.time())}.wav"
    
    try:
        # टर्बो प्रोसेसिंग: लंबी स्क्रिप्ट को तेज़ बनाने के लिए
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True # लंबे ऑडियो के लिए ज़रूरी
        )
        
        final_file = finalize_audio(output_path, remove_silence, voice_enhance)
        return final_file, f"✅ तैयार! शब्द: {len(script.split())}"
    except Exception as e:
        return None, f"❌ एरर: {str(e)}"

# शब्दों की गिनती का फंक्शन
def update_counter(text):
    count = len(text.split())
    return f"शब्दों की संख्या: {count} / 10,000"

# --- इंटरफ़ेस (Green Progress Bar Theme) ---
custom_css = """
.progress-bar { background-color: #28a745 !important; } /* हरा रंग */
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=custom_css) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Studio Pro v0.181.13**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ **वॉयस सेटिंग्स**")
            # डिफॉल्ट आवाजें और अपलोड की गई आवाज यहाँ दिखेगी
            voice_in = gr.Audio(label="आवाज़ चुनें या अपलोड करें (Joanne/Reginald)", type="filepath")
            
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited"], value="Neutral", label="🎭 इमोशन")
            
            speed_sl = gr.Slider(0.7, 1.4, 1.0, step=0.01, label="⏩ स्पीड")
            
            silence_btn = gr.Checkbox(label="🤫 सन्नाटा हटाना", value=True)
            enhance_btn = gr.Checkbox(label="✨ आवाज़ निखारना", value=True)
            
            gen_btn = gr.Button("🚀 GENERATE (TURBO GREEN)", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🪄 **स्मार्ट हुक एडिटर v3**")
            with gr.Row():
                hook_cat = gr.Dropdown(["सस्पेंस/डरावनी", "फैक्ट्स/ज्ञान", "कहानी/भावुक", "मोटिवेशन/जोश"], label="हुक का प्रकार")
                hook_btn = gr.Button("🪄 Add Smart Hook")
            
            word_counter = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="स्क्रिप्ट बॉक्स", lines=15)
            
            # लाइव वर्ड काउंटर और हुक बटन का काम
            script_in.change(update_counter, inputs=[script_in], outputs=[word_counter])
            hook_btn.click(smart_hook_editor, [script_in, hook_cat], script_in)
            
            gr.Markdown("### 🎧 **फाइनल आउटपुट**")
            audio_out = gr.Audio(label="सुनें और डाउनलोड करें", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt, silence_btn, enhance_btn], [audio_out, status])

demo.launch(share=True)
