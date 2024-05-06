########################
# Push Custom Pipeline #
########################
git clone https://huggingface.co/kotoba-tech/kotoba-whisper-v1.1
python pipeline/push_pipeline.py


##########################
# Evaluate Student Model #
##########################
MODEL="kotoba-tech/kotoba-whisper-v1.1"
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
    python run_short_form_eval.py -m ${MODEL} -d "${DATA}" -b 128 -p
    python run_short_form_eval.py -m ${MODEL} -d "${DATA}" -b 128 -s
    python run_short_form_eval.py -m ${MODEL} -d "${DATA}" -b 128 -p -s
done
