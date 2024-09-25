from typing import Optional, Dict
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from transformers.modeling_utils import PreTrainedModel
from transformers import pipeline, AutoTokenizer


class CascadedS2TTranslationPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(self,
                 model: "PreTrainedModel",
                 tgt_lang: str,
                 src_lang: str = None,
                 model_translation: "PreTrainedModel" = "facebook/nllb-200-1.3B",
                 chunk_length_s: int = 0,
                 **kwargs):
        self.type = "seq2seq_whisper"
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        kwargs.pop("task")
        kwargs.pop("tokenizer")
        super().__init__(model=model, task="automatic-speech-recognition", chunk_length_s=chunk_length_s, **kwargs)
        self.translation = pipeline("translation", model=model_translation, tokenizer=tokenizer, **kwargs)


    def _forward(self, model_inputs, **generate_kwargs):
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
