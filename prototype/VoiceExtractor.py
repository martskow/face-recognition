import os
import tempfile
import torch
import sys
from types import ModuleType


def bulletproof_amp_decorator(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]

    return lambda f: f


if not hasattr(torch, 'amp'):
    mock_amp = ModuleType('amp')
    torch.amp = mock_amp
    sys.modules['torch.amp'] = mock_amp

torch.amp.custom_fwd = bulletproof_amp_decorator
torch.amp.custom_bwd = bulletproof_amp_decorator

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import subprocess

class VoiceEmbeddingExtractor:
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def describe(self, audio_bytes):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
            temp_webm.write(audio_bytes)
            webm_path = temp_webm.name

        wav_path = webm_path.replace(".webm", ".wav")

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            signal, fs = torchaudio.load(wav_path)

            embedding = self.model.encode_batch(signal)
            embedding_np = embedding.squeeze().cpu().numpy()

        except subprocess.CalledProcessError:
            raise RuntimeError(
                "FFmpeg conversion failed. Make sure ffmpeg is installed in your system. "
                "Run 'brew install ffmpeg' in your Mac terminal."
            )
        finally:
            # Bezpieczne czyszczenie obu plików tymczasowych z dysku
            if os.path.exists(webm_path):
                os.remove(webm_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)

        return embedding_np