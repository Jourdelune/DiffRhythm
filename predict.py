# Prediction interface for Cog ⚙️
# https://cog.run/python

import tempfile

import torchaudio
from cog import BasePredictor, Input, Path

from infer.infer import inference
from infer.infer_utils import (
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        cfm, tokenizer, muq, vae = prepare_model(self.device)

        self.cfm = cfm
        self.tokenizer = tokenizer
        self.muq = muq
        self.vae = vae

    def predict(
        self,
        lyric: str = Input(
            description="Lyric to generate a song for, format: [00:00.00]lyrics",
            default="",
        ),
        audio_length: int = Input(
            description="Length of generated song", default=95, choices=[95]
        ),
        ref_prompt: str = Input(
            description="Prompt to use as style reference", default=""
        ),
        ref_audio_path: Path = Input(
            description="Audio to use as reference, have to be longer than 10 seconds",
            default="",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert (
            ref_prompt or ref_audio_path
        ), "either ref_prompt or ref_audio_path should be provided"
        assert not (
            ref_prompt and ref_audio_path
        ), "only one of them (ref_prompt or ref_audio_path) should be provided"

        assert audio_length in [95, 285], "Audio length must be 95 or 285"

        if audio_length == 95:
            max_frames = 2048
        elif audio_length == 285:  # current not available
            max_frames = 6144

        lrc_prompt, start_time = get_lrc_token(lyric, self.tokenizer, self.device)

        if ref_prompt:
            style_prompt = get_style_prompt(self.muq, prompt=ref_prompt)
        else:
            style_prompt = get_style_prompt(self.muq, ref_audio_path)

        negative_style_prompt = get_negative_style_prompt(self.device)

        latent_prompt = get_reference_latent(self.device, max_frames)

        generated_song = inference(
            cfm_model=self.cfm,
            vae_model=self.vae,
            cond=latent_prompt,
            text=lrc_prompt,
            duration=max_frames,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            chunked=True,
        )

        output_path = Path(tempfile.mkdtemp()) / "output.wav"
        torchaudio.save(output_path, generated_song, sample_rate=44100)

        return Path(output_path)
