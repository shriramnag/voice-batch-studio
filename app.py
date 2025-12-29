import os
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment

# Auto-accept Coqui TOS
os.environ["COQUI_TOS_AGREED"] = "1"

# Load Model
print("Loading AI Model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

def process_voice(voice_sample, s1, s2, s3, clean_voice):
    if not voice_sample: return None, None, None, None, "Error: Sample Required!"
    
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
    return generated_files[0], generated_files[1], generated_files[2], final_file, "Done!"

# UI Design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ AI Voice Batch Studio")
    with gr.Row():
        with gr.Column():
            sample = gr.Audio(label="Voice Sample", type="filepath")
            clean_opt = gr.Checkbox(label="Noise Reduction", value=True)
            btn = gr.Button("Generate & Merge", variant="primary")
        with gr.Column():
            with gr.Tabs():
                t1 = gr.Textbox(label="Part 1"); o1 = gr.Audio()
                t2 = gr.Textbox(label="Part 2"); o2 = gr.Audio()
                t3 = gr.Textbox(label="Part 3"); o3 = gr.Audio()
            final_out = gr.Audio(label="Final Merged Audio")

    btn.click(process_voice, [sample, t1, t2, t3, clean_opt], [o1, o2, o3, final_out])

demo.launch()
