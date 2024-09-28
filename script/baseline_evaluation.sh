################
# Japanese ASR #
################
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
  # baselines
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-large-v3" -d "${DATA}" -b 32
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-large-v2" -d "${DATA}" -b 32
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-large" -d "${DATA}" -b 32
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-medium" -d "${DATA}" -b 128
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-base" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-small" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "openai/whisper-tiny" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "reazon-research/reazonspeech-nemo-v2" -d "${DATA}" -b 256
  # main models
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-tiny" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-small" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-medium" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-large" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "japanese-asr/distil-whisper-large-v3-ja-reazonspeech-all" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v1.0" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v2.0" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 256 -p
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 256 -s
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 256 -p -s
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v2.1" -d "${DATA}" -b 256
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v2.1" -d "${DATA}" -b 256 -p
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v2.1" -d "${DATA}" -b 256 -s
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "kotoba-tech/kotoba-whisper-v2.1" -d "${DATA}" -b 256 -p -s
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "japanese-asr/distil-whisper-bilingual-v1.0" -d "${DATA}" -b 256 -p -s
done

###############
# English ASR #
###############
for DATA_CONFIG in "ami" "earnings22" "tedlium" "voxpopuli32"
do
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-large-v3" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 32 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-large-v2" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 32 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-large" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 32 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-medium" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 128 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-base" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 256 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-small" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 256 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-tiny" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 256 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "distil-whisper/distil-large-v3" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 256 --dataset-split "validation" --column-text "text"
  python run_short_form_eval.py -l "en" -t "transcribe" -m "japanese-asr/distil-whisper-bilingual-v1.0" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" -b 256 --dataset-split "validation" --column-text "text"
done
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-large-v3" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 32 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-large-v2" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 32 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-large" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 32 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-medium" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 128 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-base" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 256 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-small" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 256 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "openai/whisper-tiny" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 256 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "distil-whisper/distil-large-v3" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 256 --dataset-split "validation.clean" --column-text "text"
python run_short_form_eval.py -l "en" -t "transcribe" -m "japanese-asr/distil-whisper-bilingual-v1.0" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" -b 256 --dataset-split "validation.clean" --column-text "text"


#######################
# Translation (JA2EN) #
#######################
for DATA_CONFIG in "fleurs" "covost2"
do
  python run_short_form_eval.py -l "en" -t "translate" -m "japanese-asr/distil-whisper-bilingual-v1.0" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
  python run_short_form_eval.py -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-3.3B"
  python run_short_form_eval.py -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-1.3B"
  python run_short_form_eval.py -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-distilled-1.3B"
  python run_short_form_eval.py -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-distilled-600M"
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-large-v3" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-large-v2" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-large" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-medium" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 64
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-base" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-small" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
  python run_short_form_eval.py -l "en" -t "translate" -m "openai/whisper-tiny" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
done

#######################
# Translation (EN2JA) #
#######################
for DATA_CONFIG in "fleurs" "covost2"
do
  python run_short_form_eval.py -l "ja" -t "translate" -m "japanese-asr/distil-whisper-bilingual-v1.0" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
  python run_short_form_eval.py -l "ja" -t "translate" -m "japanese-asr/en-cascaded-s2t-translation" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-3.3B"
  python run_short_form_eval.py -l "ja" -t "translate" -m "japanese-asr/en-cascaded-s2t-translation" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-1.3B"
  python run_short_form_eval.py -l "ja" -t "translate" -m "japanese-asr/en-cascaded-s2t-translation" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-distilled-1.3B"
  python run_short_form_eval.py -l "ja" -t "translate" -m "japanese-asr/en-cascaded-s2t-translation" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32 --translation-model "facebook/nllb-200-distilled-600M"
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-large-v3" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-large-v2" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-large" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 32
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-medium" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 64
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-base" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-small" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
  python run_short_form_eval.py -l "ja" -t "translate" -m "openai/whisper-tiny" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
done