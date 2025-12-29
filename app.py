import os
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment

# Coqui TOS ko auto-accept karein
os.environ["COQUI_TOS_AGREED"] = "1"

# AI Model load karne ka function
print("AI Model load ho raha hai...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

def voice_studio(voice_sample, s1, s2, s3, clean_voice):
    if not voice_sample:
        return None, None, None, None, "Error: Voice sample upload karein!"
    
    scripts = [s1, s2, s3]
    generated_files = []
    
    for i, text in enumerate(scripts):
        if text and text.strip():
            fname = f"part_{i+1}.wav"
            tts.tts_to_file(text=text, speaker_wav=voice_sample, language="hi", file_path=fname)
            generated_files.append(fname)
        else:
            generated_files.append(None)

    combined = AudioSegment.empty()
    for f in generated_files:
        if f:
            segment = AudioSegment.from_wav(f)
            if clean_voice: segment = segment.normalize()
            combined += segment

    final_file = "final_output.wav"
    combined.export(final_file, format="wav")
    return generated_files[0], generated_files[1], generated_files[2], final_file, "Saffalta: Voice merge ho gayi!"

# Modern UI Design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ AI Voice Batch Studio Pro")
    with gr.Row():
        with gr.Column():
            sample = gr.Audio(label="Voice Clone Sample", type="filepath")
            clean_opt = gr.Checkbox(label="Enable Voice Cleaning", value=True)
            btn = gr.Button("🔥 Generate & Merge All", variant="primary")
            status = gr.Textbox(label="Status")
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Part 1"):
                    t1 = gr.Textbox(label="Script 1"); o1 = gr.Audio()
                with gr.TabItem("Part 2"):
                    t2 = gr.Textbox(label="Script 2"); o2 = gr.Audio()
                with gr.TabItem("Part 3"):
                    t3 = gr.Textbox(label="Script 3"); o3 = gr.Audio()
            final_out = gr.Audio(label="Final Merged Audio")

    btn.click(voice_studio, [sample, t1, t2, t3, clean_opt], [o1, o2, o3, final_out, status])

# Public link ke liye share=True zaroori hai
demo.launch(share=True)
