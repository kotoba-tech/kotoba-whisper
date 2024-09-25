"""https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

wget https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac -O sample_ja.flac
wget https://huggingface.co/datasets/japanese-asr/en_asr.esb_eval/resolve/main/sample.wav -O sample_en.wav
"""
from en_cascaded_s2t_translation import EnCascadedS2TTranslationPipeline
from ja_cascaded_s2t_translation import JaCascadedS2TTranslationPipeline
from transformers.pipelines import PIPELINE_REGISTRY, pipeline
from transformers import WhisperForConditionalGeneration, TFWhisperForConditionalGeneration


PIPELINE_REGISTRY.register_pipeline(
    "en-cascaded-s2t-translation",
    pipeline_class=EnCascadedS2TTranslationPipeline,
    pt_model=WhisperForConditionalGeneration,
    tf_model=TFWhisperForConditionalGeneration
)
pipe = pipeline(
    "en-cascaded-s2t-translation",
    model="distil-whisper/distil-large-v3",
    model_translation="facebook/nllb-200-distilled-600M",
    tgt_lang="jpn_Jpan",
    chunk_length_s=15,
    device_map="auto"
)
pipe.push_to_hub("japanese-asr/en-cascaded-s2t-translation")

PIPELINE_REGISTRY.register_pipeline(
    "ja-cascaded-s2t-translation",
    pipeline_class=JaCascadedS2TTranslationPipeline,
    pt_model=WhisperForConditionalGeneration,
    tf_model=TFWhisperForConditionalGeneration
)
pipe = pipeline(
    "ja-cascaded-s2t-translation",
    model="kotoba-tech/kotoba-whisper-v2.0",
    model_translation="facebook/nllb-200-distilled-600M",
    tgt_lang="eng_Latn",
    chunk_length_s=15,
    device_map="auto"
)
pipe.push_to_hub("japanese-asr/ja-cascaded-s2t-translation")
