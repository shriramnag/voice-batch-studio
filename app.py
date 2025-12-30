# Version 0.181.14-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time
import shutil
from pydub import AudioSegment, effects

# नियमों को स्वीकार करना
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# आवाज़ों को सुरक्षित रखने के लिए फोल्डर
VOICE_DIR = "custom_voices"
os.makedirs(VOICE_DIR, exist_ok=True)

# सिस्टम की डिफ़ॉल्ट आवाज़ें (इन्हें ऐप फोल्डर में होना चाहिए)
# अगर फाइल नहीं है तो एरर न आए इसके लिए चेक
def get_all_voices():
    voices = [f for f in os.listdir(VOICE_DIR) if f.endswith('.wav')]
    return ["Joanne.wav", "Reginald voice.wav"] + voices

print(f"🚀 टर्बो मोड चालू: {device} | वर्शन: 0.181.14")

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"मॉडल लोड एरर: {e}")

# --- नया आवाज़ सेव करने वाला फंक्शन ---
def save_new_voice(file):
    if file is None: return gr.update()
    filename = os.path.basename(file.name)
    dest = os.path.join(VOICE_DIR, filename)
    shutil.copy(file.name, dest)
    return gr.update(choices=get_all_voices(), value=filename)

def generate_voice(voice_name, script, emotion, speed, language, remove_silence, voice_enhance):
    if not voice_name or not script:
        return None, "❌ कृपया आवाज़ चुनें और स्क्रिप्ट डालें!"
    
    # सही रास्ता (Path) चुनना
    if voice_name in ["Joanne.wav", "Reginald voice.wav"]:
        voice_path = voice_name # ये फाइलें मेन फोल्डर में होनी चाहिए
    else:
        voice_path = os.path.join(VOICE_DIR, voice_name)

    if not os.path.exists(voice_path):
        return None, f"❌ आवाज़ फाइल नहीं मिली: {voice_name}"

    clean_text = script.replace("\n", " ").strip()
    output_path = f"vbs_output_{int(time.time())}.wav"
    
    try:
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=voice_path,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True
        )
        
        # आवाज़ निखारना और सन्नाटा हटाना
        audio = AudioSegment.from_wav(output_path)
        if remove_silence: audio = effects.strip_silence(audio, silence_thresh=-45, padding=150)
        if voice_enhance: audio = effects.normalize(audio)
        audio.export(output_path, format="wav")
        
        return output_path, "✅ ऑडियो सफलतापूर्वक तैयार है!"
    except Exception as e:
        return None, f"❌ AI Error: {str(e)}"

# --- इंटरफ़ेस ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Studio Pro v0.181.14**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ **वॉयस मेमोरी सिस्टम**")
            
            # आवाज़ चुनने की लिस्ट
            voice_select = gr.Dropdown(choices=get_all_voices(), label="मौजूदा आवाज़ चुनें", value="Joanne.wav")
            
            # नई आवाज़ अपलोड करने का बटन (जो लाइब्रेरी में सेव हो जाएगी)
            new_voice_upload = gr.File(label="नई आवाज़ को लाइब्रेरी में जोड़ें", file_types=[".wav"])
            new_voice_upload.change(save_new_voice, inputs=[new_voice_upload], outputs=[voice_select])

            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited"], value="Neutral", label="🎭 इमोशन")
            
            speed_sl = gr.Slider(0.7, 1.4, 1.0, step=0.01, label="⏩ स्पीड")
            silence_btn = gr.Checkbox(label="🤫 सन्नाटा हटाना", value=True)
            enhance_btn = gr.Checkbox(label="✨ आवाज़ निखारना", value=True)
            
            gen_btn = gr.Button("🚀 GENERATE AUDIO", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            word_counter = gr.Markdown("शब्दों की संख्या: 0 / 10,000")
            script_in = gr.Textbox(label="स्क्रिप्ट बॉक्स", lines=18)
            script_in.change(lambda x: f"शब्दों की संख्या: {len(x.split())} / 10,000", inputs=[script_in], outputs=[word_counter])
            
            audio_out = gr.Audio(label="फाइनल आउटपुट", type="filepath")

    gen_btn.click(generate_voice, [voice_select, script_in, emotion_opt, speed_sl, lang_opt, silence_btn, enhance_btn], [audio_out, status])

demo.launch(share=True)
