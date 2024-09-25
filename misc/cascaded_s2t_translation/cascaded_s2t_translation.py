from typing import Optional, Dict
import requests

import torch
import numpy as np

from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline, chunk_iter
from transformers.utils import is_torchaudio_available
from transformers.modeling_utils import PreTrainedModel
from transformers import pipeline


class CascadedS2TTranslationPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(self,
                 model: "PreTrainedModel",
                 tgt_lang: str,
                 src_lang: str = None,
                 model_translation: "PreTrainedModel" = "facebook/nllb-200-1.3B",
                 chunk_length_s: int = 0,
                 **kwargs):
        kwargs.pop("task")
        self.type = "seq2seq_whisper"
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        self.translation = pipeline("translation", model=model_translation, **kwargs)
        super().__init__(model=model, task="automatic-speech-recognition", chunk_length_s=chunk_length_s, **kwargs)

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        stride = None
        extra = {}
        if isinstance(inputs, dict):
            stride = inputs.pop("stride", None)
            # Accepting `"array"` which is the key defined in `datasets` for better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to AutomaticSpeechRecognitionPipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            extra = inputs
            inputs = _inputs
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AutomaticSpeechRecognitionPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()
                ratio = self.feature_extractor.sampling_rate / in_sampling_rate
            else:
                ratio = 1
            if stride is not None:
                if stride[0] + stride[1] > inputs.shape[0]:
                    raise ValueError("Stride is too large for input")

                # Stride needs to get the chunk length here, it's going to get
                # swallowed by the `feature_extractor` later, and then batching
                # can add extra data in the inputs, so we need to keep track
                # of the original length in the stride so we can cut properly.
                stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            # XXX: Carefuly, this variable will not exist in `seq2seq` setting.
            # Currently chunking is not possible at this level for `seq2seq` so
            # it's ok.
            align_to = getattr(self.model.config, "inputs_to_logits_ratio", 1)
            chunk_len = int(round(chunk_length_s * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_left = int(round(stride_length_s[0] * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_right = int(round(stride_length_s[1] * self.feature_extractor.sampling_rate / align_to) * align_to)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            for item in chunk_iter(
                    inputs, self.feature_extractor, chunk_len, stride_left, stride_right, self.torch_dtype
            ):
                item["audio_array"] = inputs
                yield item
        else:
            if inputs.shape[0] > self.feature_extractor.n_samples:
                processed = self.feature_extractor(
                    inputs,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    truncation=False,
                    padding="longest",
                    return_tensors="pt",
                )
            else:
                processed = self.feature_extractor(
                    inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
                )

            if self.torch_dtype is not None:
                processed = processed.to(dtype=self.torch_dtype)
            if stride is not None:
                processed["stride"] = stride
            yield {"is_last": True, "audio_array": inputs, **processed, **extra}

    def _forward(self, model_inputs, **generate_kwargs):
        if "tgt_lang" in generate_kwargs:
            self.tgt_lang = generate_kwargs.pop("tgt_lang")
        if "src_lang" in generate_kwargs:
            self.src_lang = generate_kwargs.pop("src_lang")
        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        is_last = model_inputs.pop("is_last")
        encoder = self.model.get_encoder()
        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # custom processing for Whisper timestamps and word-level timestamps
        if inputs.shape[-1] > self.feature_extractor.nb_max_frames:
            generate_kwargs["input_features"] = inputs
        else:
            generate_kwargs["encoder_outputs"] = encoder(inputs, attention_mask=attention_mask)

        tokens = self.model.generate(attention_mask=attention_mask, **generate_kwargs)
        # whisper longform generation stores timestamps in "segments"
        out = {"tokens": tokens}
        if self.type == "seq2seq_whisper":
            if stride is not None:
                out["stride"] = stride
        return {"is_last": is_last, **out, **model_inputs}

    def postprocess(self,
                    model_outputs,
                    decoder_kwargs: Optional[Dict] = None,
                    return_timestamps=None,
                    return_language=None):
        assert len(model_outputs) > 0
        outputs = super().postprocess(
            model_outputs=model_outputs,
            decoder_kwargs=decoder_kwargs,
            return_language=self.src_lang
        )
        chunks = outputs.pop("chunks")
        outputs["text_asr"] = "".join([c["text"] for c in chunks])
        outputs["text"] = self.translation(outputs["text_asr"], src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        return outputs
