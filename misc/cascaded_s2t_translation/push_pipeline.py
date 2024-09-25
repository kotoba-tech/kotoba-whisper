from cascaded_s2t_translation import CascadedS2TTranslationPipeline
from transformers.pipelines import PIPELINE_REGISTRY, pipeline
from transformers import WhisperForConditionalGeneration, TFWhisperForConditionalGeneration


model_alias = "kotoba-tech/kotoba-whisper-v1.1"
PIPELINE_REGISTRY.register_pipeline(
    "cascaded-s2t-translation",
    pipeline_class=CascadedS2TTranslationPipeline,
    pt_model=WhisperForConditionalGeneration,
    tf_model=TFWhisperForConditionalGeneration
)
pipe = pipeline(
    "cascaded-s2t-translation",
    model="distil-whisper/distil-large-v3",
    model_translation="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="jpn_Jpan",
    chunk_length_s=15,
    device_map="auto"
)
pipe.push_to_hub("japanese-asr/cascaded-s2t-translation")
