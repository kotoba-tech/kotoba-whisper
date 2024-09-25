################
# Japanese ASR #
################
for MODEL in "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "openai/whisper-medium" "openai/whisper-small" "openai/whisper-base" "openai/whisper-tiny"
do
  for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
  do
    python run_short_form_eval.py -l "ja" -t "transcribe" -m ${MODEL} -d "${DATA}" -b 32
  done
done

for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
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
done

###############
# English ASR #
###############
for MODEL in "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "openai/whisper-medium" "openai/whisper-small" "openai/whisper-base" "openai/whisper-tiny" "distil-whisper/distil-large-v3"
do
  python run_short_form_eval.py -l "en" -t "transcribe" -m "${MODEL}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "ami" --dataset-split "validation" --column-text "text" -b 32
  python run_short_form_eval.py -l "en" -t "transcribe" -m "${MODEL}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "earnings22" --dataset-split "validation" --column-text "text" -b 32
  python run_short_form_eval.py -l "en" -t "transcribe" -m "${MODEL}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "tedlium" --dataset-split "validation" --column-text "text" -b 32
  python run_short_form_eval.py -l "en" -t "transcribe" -m "${MODEL}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" --dataset-split "validation.clean" --column-text "text" -b 32
  python run_short_form_eval.py -l "en" -t "transcribe" -m "${MODEL}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "voxpopuli" --dataset-split "validation" --column-text "text" -b 32
done

python run_short_form_eval.py -o "eval_pipeline_tmp" -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "ami" --dataset-split "validation" --column-text "text" -b 256
python run_short_form_eval.py -o "eval_pipeline_tmp" -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "earnings22" --dataset-split "validation" --column-text "text" -b 256
python run_short_form_eval.py -o "eval_pipeline_tmp" -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "tedlium" --dataset-split "validation" --column-text "text" -b 256
python run_short_form_eval.py -o "eval_pipeline_tmp" -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" --dataset-split "validation.clean" --column-text "text" -b 256
python run_short_form_eval.py -o "eval_pipeline_tmp" -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "voxpopuli" --dataset-split "validation" --column-text "text" -b 256
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
  python run_short_form_eval.py -o "eval_pipeline_tmp" -l "ja" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "${DATA}" -b 256
done



for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "${DATA}" -b 256
done
python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "ami" --dataset-split "validation" --column-text "text" -b 256
python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "earnings22" --dataset-split "validation" --column-text "text" -b 256
python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "tedlium" --dataset-split "validation" --column-text "text" -b 256
python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" --dataset-split "validation.clean" --column-text "text" -b 256
python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "voxpopuli" --dataset-split "validation" --column-text "text" -b 256


# Translation (JA2EN)
for DATA_CONFIG in "fleurs" "covost2"
do
  python run_short_form_eval.py -l "en" -t "translate" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
done
# Translation (EN2JA)
for DATA_CONFIG in "fleurs" "covost2"
do
  python run_short_form_eval.py -l "ja" -t "translate" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
done