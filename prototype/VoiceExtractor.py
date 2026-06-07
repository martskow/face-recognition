import sys
from types import ModuleType
import io
import numpy as np
import torch


# =========================================================================
def bulletproof_amp_decorator(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f


if not hasattr(torch, 'amp'):  # pragma: no cover
    mock_amp = ModuleType('amp')  # pragma: no cover
    torch.amp = mock_amp  # pragma: no cover
    sys.modules['torch.amp'] = mock_amp  # pragma: no cover

torch.amp.custom_fwd = bulletproof_amp_decorator
torch.amp.custom_bwd = bulletproof_amp_decorator
# =========================================================================

import av
from speechbrain.inference.speaker import EncoderClassifier

class VoiceEmbeddingExtractor:
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )

    def describe(self, audio_bytes):
        try:
            audio_stream = io.BytesIO(audio_bytes)
            container = av.open(audio_stream)
            stream = container.streams.audio[0]

            # Resampler w locie: format float, 1 kanał (mono), 16000 Hz
            # Czyli dokładnie to, czego wymaga model SpeechBrain (ECAPA-TDNN)
            resampler = av.AudioResampler(format='flt', layout='mono', rate=16000)

            audio_segments = []

            for packet in container.decode(stream):
                for frame in resampler.resample(packet):
                    audio_segments.append(frame.to_ndarray().flatten())

            for frame in resampler.resample(None):
                audio_segments.append(frame.to_ndarray().flatten())

            if not audio_segments:
                raise ValueError("Could not decode any audio frames from the provided data.")

            signal_np = np.concatenate(audio_segments)

            signal_tensor = torch.from_numpy(signal_np).float().unsqueeze(0)

            embedding = self.model.encode_batch(signal_tensor)
            embedding_np = embedding.squeeze().cpu().numpy()

            return embedding_np

        except Exception as e:
            raise RuntimeError(f"Voice processing error (PyAV): {e}")