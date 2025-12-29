import os
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment

# नियम स्वीकार करें
os.environ["COQUI_TOS_AGREED"] = "1"

# AI मॉडल लोड करें
print("AI Model load ho raha hai...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

def voice_studio(sample, s1, s2, s3, clean):
    scripts = [s1, s2, s3]
    files = []
    for i, text in enumerate(scripts):
        if text and text.strip():
            path = f"part_{i}.wav"
            tts.tts_to_file(text=text, speaker_wav=sample, language="hi", file_path=path)
            files.append(path)
    
    combined = AudioSegment.empty()
    for f in files:
        seg = AudioSegment.from_wav(f)
        if clean: seg = seg.normalize()
        combined += seg
        
    combined.export("final_audio.wav", format="wav")
    return "final_audio.wav"

# UI डिजाइन
demo = gr.Interface(
    fn=voice_studio,
    inputs=[gr.Audio(type="filepath"), "text", "text", "text", "checkbox"],
    outputs="audio",
    title="VoiceBatch Studio Pro"
)

if __name__ == "__main__":
    demo.launch(share=True)
