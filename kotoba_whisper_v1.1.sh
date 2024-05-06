########################
# Push Custom Pipeline #
########################
git clone https://huggingface.co/kotoba-tech/kotoba-whisper-v1.1
python pipeline/push_pipeline.py


##########################
# Evaluate Student Model #
##########################
for MODEL in "kotoba-tech/kotoba-whisper-v1.0" "kotoba-tech/kotoba-whisper-v1.1"
do
    for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
    do
        python run_eval_pipeline.py -m ${MODEL} -d "${DATA}" -b 32
        python run_eval_pipeline.py -m ${MODEL} -d "${DATA}" -b 32 -p
        python run_eval_pipeline.py -m ${MODEL} -d "${DATA}" -b 32 -s
        python run_eval_pipeline.py -m ${MODEL} -d "${DATA}" -b 32 -p -s
    done
done
