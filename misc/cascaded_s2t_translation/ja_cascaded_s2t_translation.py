from typing import Optional, Dict
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from transformers.modeling_utils import PreTrainedModel
from transformers import pipeline, AutoTokenizer


class JaCascadedS2TTranslationPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(self,
                 model: "PreTrainedModel",
                 tgt_lang: str,
                 model_translation: "PreTrainedModel" = "facebook/nllb-200-1.3B",
                 chunk_length_s: int = 0,
                 **kwargs):
        self.tgt_lang = tgt_lang
        kwargs.pop("task")
        super().__init__(model=model, task="automatic-speech-recognition", chunk_length_s=chunk_length_s, **kwargs)
        kwargs["tokenizer"] = AutoTokenizer.from_pretrained(model_translation)
        self.translation = pipeline("translation", model=model_translation, **kwargs)

    def _forward(self, model_inputs, **generate_kwargs):
        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        is_last = model_inputs.pop("is_last")
        encoder = self.model.get_encoder()
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
        if stride is not None:
            return {"is_last": is_last, "stride": stride, "tokens": tokens, **model_inputs}
        return {"is_last": is_last, "tokens": tokens, **model_inputs}

    def postprocess(self, model_outputs, decoder_kwargs: Optional[Dict] = None, **kwargs):
        outputs = super().postprocess(model_outputs=model_outputs, decoder_kwargs=decoder_kwargs)
        trans = self.translation(outputs["text"], src_lang="jpn_Jpan", tgt_lang=self.tgt_lang)[0]['translation_text']
        return {"text": trans, "text_asr": outputs["text"]}
