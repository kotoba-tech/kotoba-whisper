# kotoba-whisper-v1.1 is based on kotoba-whisper-v1.0 but with post-processing module built in the huggingface pipeline.
########################
# Push Custom Pipeline #
########################
git clone https://huggingface.co/kotoba-tech/kotoba-whisper-v2.1
python pipeline/push_pipeline.py


##########################
# Evaluate Student Model #
##########################
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
    python run_short_form_eval.py -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 512
    python run_short_form_eval.py -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 512 -p
    python run_short_form_eval.py -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 512 -s
    python run_short_form_eval.py -m "kotoba-tech/kotoba-whisper-v1.1" -d "${DATA}" -b 512 -p -s
done
