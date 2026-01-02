import torch
import torchaudio
from .models.tts import TTS

def run_tts(text, reference_audio_path, output_path="output.wav"):
    """
    यह मुख्य फंक्शन है जो टेक्स्ट को आवाज में बदलेगा।
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. मॉडल लोड करें
    model = TTS(device=device)

    # 2. रेफरेंस ऑडियो लोड करें (जिसकी आवाज कॉपी करनी है)
    ref_wav, sr = torchaudio.load(reference_audio_path)
    
    # 3. आवाज जनरेट करें
    print("Generating audio...")
    generated_wav = model.synthesize(text, ref_wav, sr)

    # 4. फाइल सेव करें
    torchaudio.save(output_path, generated_wav.cpu(), 24000)
    print(f"Success! Audio saved at: {output_path}")

if __name__ == "__main__":
    # टेस्ट करने के लिए
    run_tts("नमस्ते, यह एक टेस्ट आवाज है।", "ref.wav")
