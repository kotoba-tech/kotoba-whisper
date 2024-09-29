for DURATION in 10 30 60 300
do
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/distil-whisper-bilingual-v1.0" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-3.3B"
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-1.3B"
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-1.3B"
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-600M"
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v3" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v2" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-medium" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-base" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-small" -n 10
  python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-tiny" -n 10
done

for DURATION in 10 30 60 300
do
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/distil-whisper-bilingual-v1.0" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-3.3B"
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-1.3B"
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-1.3B"
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-600M"
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v3" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v2" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-medium" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-base" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-small" -n 10
  python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-tiny" -n 10
done


DURATION=300
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/distil-whisper-bilingual-v1.0" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-3.3B"
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-1.3B"
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-1.3B"
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-600M"
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v3" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v2" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-medium" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-base" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-small" -n 10
python run_speed_eval.py -a "sdpa" -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-tiny" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/distil-whisper-bilingual-v1.0" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-3.3B"
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-1.3B"
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-1.3B"
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "japanese-asr/ja-cascaded-s2t-translation" -n 10 --translation-model "facebook/nllb-200-distilled-600M"
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v3" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large-v2" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-large" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-medium" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-base" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-small" -n 10
python run_speed_eval.py -d ${DURATION} -l "en" -t "translate" -m "openai/whisper-tiny" -n 10