import torch
import sounddevice as sd
import soundfile as sf
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "mps" if torch.backends.mps.is_available() else "cpu"



model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map=device,
    dtype=torch.float32,     # <- important
    attn_implementation="eager",
)

ref_audio = "Converted.wav"
ref_text  = "This is a bunch of scripts intended to be used by anyone to find stuff without remembering the formula and other things"

wavs, sr = model.generate_voice_clone(
    text=input('Enter Test Here: '),
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

sf.write("OutputVoiceA.wav", wavs[0], sr)
sd.play(wavs[0], sr)
sd.wait(
