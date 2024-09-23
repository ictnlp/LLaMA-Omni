# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess
import json
import soundfile as sf
import torch

from cog import BasePredictor, Input, Path, BaseModel
from fairseq import utils as fairseq_utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.infer.infer import create_data_loader, ctc_postprocess


MODEL_CACHE = "models"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/ictnlp/LLaMA-Omni/{MODEL_CACHE}.tar"
)
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


class ModelOutput(BaseModel):
    audio: Path
    text: str


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Model
        disable_torch_init()
        self.tokenizer, self.model, _ = load_pretrained_model(
            f"{MODEL_CACHE}/Llama-3.1-8B-Omni", model_base=None, s2s=True
        )

        with open(f"{MODEL_CACHE}/vocoder/config.json") as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(
            f"{MODEL_CACHE}/vocoder/g_00500000", vocoder_cfg
        ).cuda()

    def predict(
        self,
        input_audio: Path = Input(description="Input audio"),
        prompt: str = Input(
            default="Please directly answer the questions in the user's speech"
        ),
        temperature: float = Input(
            description="Controls randomness. Lower values make the model more deterministic, higher values make it more random.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        top_p: float = Input(
            description="Controls diversity of the output. Valid when temperature > 0. Lower values make the output more focused, higher values make it more diverse.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate", default=256, ge=1
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        questions = [
            {
                "speech": str(input_audio),
                "conversations": [{"from": "human", "value": f"<speech>\n{prompt}"}],
            }
        ]

        data_loader = create_data_loader(
            questions,
            self.tokenizer,
            self.model.config,
            input_type="mel",
            mel_size=128,
            conv_mode="llama_3",
        )

        (input_ids, speech_tensor, speech_length) = next(iter(data_loader))

        input_ids = input_ids.to(device="cuda", non_blocking=True)
        speech_tensor = speech_tensor.to(
            dtype=torch.float16, device="cuda", non_blocking=True
        )
        speech_length = speech_length.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids, output_units = self.model.generate(
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_length,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p if temperature > 0 else None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=128004,
                streaming_unit_gen=False,
            )

        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        output_units = ctc_postprocess(
            output_units, blank=self.model.config.unit_vocab_size
        )

        print(prediction)

        print(f"output_units: {output_units}")
        print(type(output_units))

        output_units = [(list(map(int, output_units.strip().split())))]

        x = {
            "code": torch.LongTensor(output_units[0]).view(1, -1),
        }

        x = fairseq_utils.move_to_cuda(x)
        wav = self.vocoder(x, True)

        out_path = "/tmp/out.wav"

        sf.write(
            out_path,
            wav.detach().cpu().numpy(),
            16000,
        )

        return ModelOutput(audio=Path(out_path), text=prediction)
