# kotoba-whisper-v2 distillation version
##########
# Config #
##########
WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
HF_MODEL_ALIAS="distil-whisper-bilingual"  # Model alias used when pushing models.
#WARMUP_STEPS=500  # Warmup step.
huggingface-cli login  # Configure huggingface.
accelerate config  # Configure accelerate.

################################
# Preprocess Dataset (English) #
################################
process_en_main () {
  DATASET_CONFIG=${1}
  NUM_PROC=${2}
  BATCH=${3}
  accelerate launch --multi_gpu run_pseudo_labelling_v2.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --attn_implementation "sdpa" \
    --dataset_name "japanese-asr/en_asr.mls" \
    --dataset_split "train" \
    --num_chunks 15 \
    --text_column_name "transcription,transcription/ja_gpt3.5" \
    --language "en,ja" \
    --task "transcribe,translate" \
    --dataset_config_name "${DATASET_CONFIG}" \
    --per_device_eval_batch_size "${BATCH}" \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers ${NUM_PROC} \
    --logging_steps 5000 \
    --max_label_length 128 \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output.en_asr.mls__${DATASET_CONFIG}" \
    --hub_model_id "${HF_ORG}/whisper_transcriptions.mls"
  rm -rf "output.en_asr.mls__${DATASET_CONFIG}"
  rm -rf "${HOME}/.cache/huggingface/datasets/japanese-asr___en_asr.mls/${DATASET_CONFIG}"
}

for i in {0..9}
do
  process_en_main "subset_${i}" 8 32
done

# fix config name
wget https://huggingface.co/datasets/${HF_ORG}/whisper_transcriptions.mls/raw/main/README.md
```python
import re

with open('README.md', 'r') as f:
    config = f.read()
config_ids = sorted(set(re.findall(r"config_name: subset_[\S]*\n", config)))
for n, i in enumerate(config_ids):
  config = config.replace(i, f"config_name: subset_{n}\n")
with open("README.md", "w") as f:
  f.write(config)
```


#################################
# Preprocess Dataset (Japanese) #
#################################
process_ja_main () {
  DATASET_CONFIG=${1}
  NUM_PROC=${2}
  BATCH=${3}
  python -c """from datasets import load_dataset; load_dataset('japanese-asr/ja_asr.reazon_speech_all', '"${DATASET_CONFIG}"', num_proc=16)"""
  accelerate launch --multi_gpu run_pseudo_labelling_v2.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --attn_implementation "sdpa" \
    --dataset_name "japanese-asr/ja_asr.reazon_speech_all" \
    --dataset_split "train" \
    --num_chunks 15 \
    --text_column_name "transcription,transcription/en_gpt3.5" \
    --language "ja,en" \
    --task "transcribe,translate" \
    --dataset_config_name "${DATASET_CONFIG}" \
    --per_device_eval_batch_size "${BATCH}" \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers ${NUM_PROC} \
    --logging_steps 5000 \
    --max_label_length 128 \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output.ja_asr.reazon__${DATASET_CONFIG}" \
    --hub_model_id "${HF_ORG}/whisper_transcriptions.reazon_speech_all"
}

for i in {0..15}
do
  process_ja_main "subset_${i}" 8 32
done

# fix config name
wget "https://huggingface.co/datasets/${HF_ORG}/whisper_transcriptions.reazon_speech_all/raw/main/README.md"

```python
import re

with open('README.md', 'r') as f:
    config = f.read()
config_ids = sorted(set(re.findall(r"config_name: subset_[\S]*\n", config)))
for n, i in enumerate(config_ids):
  config = config.replace(i, f"config_name: subset_{n}\n")
with open("README.md", "w") as f:
  f.write(config)
```


#####################
# Filtering Dataset #
#####################
# English
filter_en () {
  DATASET_CHUNK_ID=${1}
  python run_data_filtering_v2.py \
    -d "${HF_ORG}/whisper_transcriptions.mls" \
    --dataset_config_name "subset_${DATASET_CHUNK_ID}" \
    --task_filtering "transcribe" \
    --language_filtering "en" \
    --task "transcribe,translate,transcribe,translate" \
    --language "en,ja,en,ja" \
    --text_column_name "transcription,transcription/ja_gpt3.5,whisper_transcription,whisper_transcription/ja_gpt3.5" \
    --text_column_prediction "whisper_transcription" \
    --text_column_label "transcription" \
    --wer_threshold ${WER_THRESHOLD} \
    --preprocessing_num_workers 64 \
    --preprocessing_batch_size 64 \
    --skip_logmel
  python run_data_filtering_v2.py \
    -d "${HF_ORG}/whisper_transcriptions.mls.wer_${WER_THRESHOLD}" \
    --dataset_config_name "subset_${DATASET_CHUNK_ID}" \
    --task_filtering "transcribe" \
    --language_filtering "en" \
    --task "transcribe,translate,transcribe,translate" \
    --language "en,ja,en,ja" \
    --text_column_name "transcription,transcription/ja_gpt3.5,whisper_transcription,whisper_transcription/ja_gpt3.5" \
    --text_column_prediction "whisper_transcription" \
    --text_column_label "transcription" \
    --wer_threshold ${WER_THRESHOLD} \
    --preprocessing_num_workers 64 \
    --preprocessing_batch_size 64 \
    --skip_filtering
  rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___whisper_transcriptions.mls/subset_${DATASET_CHUNK_ID}"
  rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___whisper_transcriptions.mls.wer_${WER_THRESHOLD}/subset_${DATASET_CHUNK_ID}"
  rm -rf "${HOME}/.cache/huggingface/datasets/downloads"
}

for i in {0..138}
do
  filter_en ${i}
done


# runpod_pre_4
for i in {21..40}
do
  filter_en ${i}
done
# runpod_pre_5
for i in {41..60}
do
  filter_en ${i}
done
# runpod_pre_3
for i in {71..80}
do
  filter_en ${i}
done



filter_ja () {
  DATASET_CHUNK_ID=${1}
  python run_data_filtering_v2.py \
    -d "${HF_ORG}/whisper_transcriptions.reazon_speech_all" \
    --dataset_config_name "subset_${DATASET_CHUNK_ID}" \
    --task_filtering "transcribe" \
    --language_filtering "ja" \
    --task "transcribe,translate,transcribe,translate" \
    --language "ja,en,ja,en" \
    --text_column_name "transcription,transcription/en_gpt3.5,whisper_transcription,whisper_transcription/en_gpt3.5" \
    --text_column_prediction "whisper_transcription" \
    --text_column_label "transcription" \
    --wer_threshold ${WER_THRESHOLD} \
    --preprocessing_num_workers 64 \
    --preprocessing_batch_size 64 \
    --skip_logmel
  python run_data_filtering_v2.py \
    -d "${HF_ORG}/whisper_transcriptions.reazon_speech_all.wer_${WER_THRESHOLD}" \
    --dataset_config_name "subset_${DATASET_CHUNK_ID}" \
    --task_filtering "transcribe" \
    --language_filtering "ja" \
    --task "transcribe,translate,transcribe,translate" \
    --language "ja,en,ja,en" \
    --text_column_name "transcription,transcription/en_gpt3.5,whisper_transcription,whisper_transcription/en_gpt3.5" \
    --text_column_prediction "whisper_transcription" \
    --text_column_label "transcription" \
    --wer_threshold ${WER_THRESHOLD} \
    --preprocessing_num_workers 64 \
    --preprocessing_batch_size 64 \
    --skip_filtering
  rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___whisper_transcriptions.reazon_speech_all/subset_${DATASET_CHUNK_ID}"
  rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___whisper_transcriptions.reazon_speech_all.wer_${WER_THRESHOLD}/subset_${DATASET_CHUNK_ID}"
  rm -rf "${HOME}/.cache/huggingface/datasets/downloads"
}
# runpod_pre_1
for i in {61..70}
do
  filter_ja ${i}
done
# runpod_pre_7
for i in {53..60}
do
  filter_ja ${i}
done
# runpod_pre_8
for i in {101..110}
do
  filter_ja ${i}
done
# runpod_pre_6
for i in {91..100}
do
  filter_ja ${i}
done
# runpod_pre_9
for i in {111..121}
do
  filter_ja ${i}
done
# runpod_pre_3
for i in {81..90}
do
  filter_ja ${i}
done
# runpod_pre_2
for i in {71..80}
do
  filter_ja ${i}
done
#TODO
for i in {121..130}
do
  filter_ja ${i}
done
for i in {131..140}
do
  filter_ja ${i}
done
for i in {141..150}
do
  filter_ja ${i}
done
for i in {151..160}
do
  filter_ja ${i}
done
for i in {161..170}
do
  filter_ja ${i}
done
for i in {171..180}
do
  filter_ja ${i}
done
for i in {181..190}
do
  filter_ja ${i}
done
for i in {191..200}
do
  filter_ja ${i}
done
for i in {201..210}
do
  filter_ja ${i}
done
for i in {211..220}
do
  filter_ja ${i}
done
for i in {221..226}
do
  filter_ja ${i}
done



############################
# Initialize Student Model #
############################
huggingface-cli repo create "${HF_MODEL_ALIAS}"
git lfs install
git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
cp create_student_model.py "${HF_MODEL_ALIAS}"
cp run_distillation.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}"
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"

##########################
# Training Student Model #
##########################
export WANDB_DISABLED="false"
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized" \
  --train_dataset_config_name "${DATASET_TYPE}" \
  --language "ja" \
  --max_label_length 128 \
  --train_split_name "train" \
  --save_steps 2500 \
  --warmup_steps "${WARMUP_STEPS}" \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 50 \
  --save_total_limit 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --preprocessing_num_workers 64 \
  --dataloader_num_workers 1 \
  --output_dir "./" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --push_to_hub \
  --overwrite_output_dir \
  --num_train_epochs 8

##########################
# Evaluate Student Model #
##########################
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
  python run_short_form_eval.py -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "${DATA}" -b 512
done


####################
# Trouble Shooting #
####################
# SSL Error
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export CURL_CA_BUNDLE=''


