"""https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

wget https://datasets-server.huggingface.co/assets/japanese-asr/ja2en.s2t_translation/--/ed5e9dad649aa81ccb90cc953903222197490ddb/--/covost2/validation/0/audio/audio.mp3\?Expires\=1727275637\&Signature\=NNaajc1pfDNGIWqAuO76DCYXYPzX379jk5HNvO24sp\~DGqx2vNbkFZyJtOIde\~8YDHg2rkpnz8Z7IaE-NXd8blsR9t67OszfQg69dXNafs2LiTjDLdRXyUvIMyOpCqAI1W5rgsuOXue27\~2KNAtTiWoCEDWU13CnoVv7rM3Ge4CGk71pT3X45K6b\~XPek8tIYj6KGm0Wqe9vTKuxDFigOFIn4qGo2dOAlOJhVqnPFgue0R5rVkKcZxJhTggGtPB7XLxZ-my8LEOAgKvKsSiyPvNMXnTfc3ugiOUNGLskXtMLa0QZXLeKz-tpABn90843TRIgz-bXQP1D7VesHEu5BQ__\&Key-Pair-Id\=K3EI6M078Z3AC3 -O sample_ja.mp3
wget https://datasets-server.huggingface.co/assets/japanese-asr/en2ja.s2t_translation/--/85c78a02e054c3fa6d9926f2385ad2f7f6fa6668/--/covost2/validation/0/audio/audio.mp3?Expires=1727275761&Signature=N~O2Dh7L3VQlpn7V~1LanF8Sz8TEdr6CsOZl2EsD9E1ZGYSjWCqC3pStP9bMziuQ1wD9hs-kR5Op~BFWdZ5jx9GpfZhAsLNuUMVePn5J5WP2kCJS-KT~PQayzgK6lhx1JJWxPwiLT30jTFu9SJAchZJ9ccU3SudOv~9U2LP-Ajr-ud6TKFd0sW1awEZ3NpIVhJ5a3DwshHTS7WlPTs4g0BY050avq5zp-9pPsXT1l~BILNfyMuB4hew7f9iJ7K7p4~d3b-xRLSb4HnMiJw2vXiDk63aqU5jXVO3NA-AW54n4c1FbM9vQWXv8h-cVQ70ilzMxS~tuV3qpd7c5mM5Dmw__&Key-Pair-Id=K3EI6M078Z3AC3 -O sample_en.mp3
"""
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
)
output = pipe("./sample_en.mp3", src_lang="eng_Latn", tgt_lang="jpn_Jpan")
print(output)

# pipe.push_to_hub(model_alias)
