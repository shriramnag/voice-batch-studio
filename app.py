# Version 0.181.12-release
import os
import gradio as gr
from TTS.api import TTS
import torch
import time
from pydub import AudioSegment, effects

# नियमों को स्वीकार करना और CPU/GPU ऑप्टिमाइज़ेशन
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    torch.set_num_threads(os.cpu_count())

print(f"🚀 मोड: {device} | वर्शन: 0.181.12-release")

try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"मॉडल लोड एरर: {e}")

# --- सन्नाटा हटाने और आवाज़ निखारने का फंक्शन ---
def enhance_audio(file_path, remove_silence, enhance):
    audio = AudioSegment.from_wav(file_path)
    
    if remove_silence:
        # सन्नाटा हटाना
        audio = effects.strip_silence(audio, silence_thresh=-42, padding=100)
    
    if enhance:
        # आवाज़ को भारी और साफ़ बनाना (Studio Quality)
        audio = effects.normalize(audio)
    
    audio.export(file_path, format="wav")
    return file_path

# --- स्मार्ट स्क्रिप्ट एडिटर (Suspense Fix) ---
def smart_editor(text, style):
    if not text: return text
    
    hooks = {
        "सस्पेंस (Suspense)": "क्या आपको पता है? एक ऐसी अनसुनी कहानी जिसने पूरी दुनिया को हिला कर रख दिया... ",
        "भावुक (Emotional)": "एक ऐसी दास्ताँ जो शायद आपकी रूह को छू ले और आँखों में नमी भर दे... ",
        "जोशीला (Excited)": "नमस्कार दोस्तों! तैयार हो जाइए एक बहुत ही रोमांचक सफर पर चलने के लिए! "
    }
    
    # अब "शुरुआत:" जैसा शब्द नहीं आएगा, सीधा डायलॉग जुड़ेगा
    if style in hooks:
        return hooks[style] + text
    return text

def generate_voice(voice_sample, script, emotion, speed, language, remove_silence, voice_enhance):
    if not voice_sample or not script:
        return None, "❌ डेटा अधूरा है!"
    
    # AI Error फिक्स करने के लिए टेक्स्ट क्लीनिंग
    clean_text = script.replace("\n", " ").strip()
    output_path = f"vbs_final_{int(time.time())}.wav"
    
    try:
        start_time = time.time()
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=voice_sample,
            language=language,
            file_path=output_path,
            emotion=emotion,
            speed=speed,
            enable_text_splitting=True
        )
        
        # एक्स्ट्रा फीचर्स: सन्नाटा हटाना और आवाज़ निखारना
        final_file = enhance_audio(output_path, remove_silence, voice_enhance)
        
        duration = round(time.time() - start_time, 2)
        return final_file, f"✅ सफलता! समय: {duration}s"
    except Exception as e:
        return None, f"❌ AI Error Fix Needed: {str(e)}"

# --- इंटरफ़ेस ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ **VoiceBatch Studio Pro v0.181.12**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ **कंट्रोल पैनल**")
            voice_in = gr.Audio(label="अपना सैंपल दें", type="filepath")
            
            with gr.Row():
                lang_opt = gr.Dropdown(choices=["hi", "en"], value="hi", label="🌍 भाषा")
                emotion_opt = gr.Dropdown(choices=["Neutral", "Sad", "Happy", "Angry", "Excited"], value="Neutral", label="🎭 इमोशन")
            
            speed_sl = gr.Slider(0.7, 1.4, 1.0, step=0.01, label="⏩ स्पीड कंट्रोल")
            
            # आपकी डिमांड वाले बटन यहाँ हैं
            silence_btn = gr.Checkbox(label="🤫 सन्नाटा हटाएं (Silence Remover)", value=True)
            enhance_btn = gr.Checkbox(label="✨ आवाज़ निखारें (Voice Enhancer)", value=True)
            
            gen_btn = gr.Button("🚀 GENERATE VOICE", variant="primary")
            status = gr.Textbox(label="सिस्टम स्टेटस", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 🪄 **स्मार्ट एडिटर**")
            with gr.Row():
                script_style = gr.Dropdown(["सामान्य", "सस्पेंस (Suspense)", "भावुक (Emotional)", "जोशीला (Excited)"], value="सामान्य", label="अंदाज़")
                improve_btn = gr.Button("🪄 Improve Script")
            
            script_in = gr.Textbox(label="स्क्रिप्ट", lines=15)
            improve_btn.click(smart_editor, [script_in, script_style], script_in)
            
            gr.Markdown("### 🎧 **आउटपुट**")
            audio_out = gr.Audio(label="सुनें और डाउनलोड करें", type="filepath")

    gen_btn.click(generate_voice, [voice_in, script_in, emotion_opt, speed_sl, lang_opt, silence_btn, enhance_btn], [audio_out, status])

demo.launch(share=True)
