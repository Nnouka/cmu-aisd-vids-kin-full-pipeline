from functools import cached_property
from pathlib import Path

import scipy.io.wavfile
import torch

from app.core.config import settings


class TTSService:
    @cached_property
    def _model(self):
        from deepkin.models.flex_tts import FlexKinyaTTS

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return FlexKinyaTTS.from_pretrained(device, settings.tts_model_path)

    def preload(self) -> None:
        _ = self._model

    def is_loaded(self) -> bool:
        return "_model" in self.__dict__

    def synthesize_to_file(self, text: str, output_path: Path, speaker_id: int | None = None) -> Path:
        from deepkin.data.kinya_norm import text_to_sequence
        from deepkin.modules.tts_commons import intersperse

        output_path.parent.mkdir(parents=True, exist_ok=True)
        active_speaker = speaker_id if speaker_id is not None else settings.speaker_id

        text_id_sequence = intersperse(text_to_sequence(text, norm=True), 0)
        audio_data = self._model(text_id_sequence, active_speaker)

        audio_numpy = audio_data.cpu().numpy().squeeze()
        sampling_rate = 24000
        scipy.io.wavfile.write(str(output_path), sampling_rate, audio_numpy)
        return output_path
