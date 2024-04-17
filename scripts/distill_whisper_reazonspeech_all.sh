##########
# Config #
##########
DATASET_TYPE="all"  # Dataset type alias.
WARMUP_STEPS=500  # Warmup step.
WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"  # Dataset alias used when pushing datasets.
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"  # Model alias used when pushing models.
huggingface-cli login  # Configure huggingface.

######################
# Preprocess Dataset #
######################
CHUNK_SIZE=50
process_chunk () {
  DATASET_CHUNK_ID=${1}
  CHUNK_START=${2}
  CHUNK_END=${3}
  export PREPROCESSING_ONLY=0
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
  python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 25 -s "${CHUNK_START}" -e "${CHUNK_END}"
  accelerate launch --multi_gpu scripts/run_pseudo_labelling.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --dataset_name "${PWD}/scripts/reazonspeech_manual_dataloader.py" \
    --dataset_config_name "${DATASET_TYPE}" \
    --dataset_dir_suffix "${CHUNK_START}_${CHUNK_END}" \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 1 \
    --logging_steps 5000 \
    --max_label_length 128 \
    --language "ja" \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
    --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"
  rm -rf "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"
  rm -rf "${HOME}/.cache/reazon_manual_download/all_${CHUNK_START}_${CHUNK_END}"
  rm -rf "${HOME}/.cache/huggingface/datasets/reazonspeech_manual_dataloader/all-dataset_dir_suffix=${CHUNK_START}_${CHUNK_END}"
}
process_chunk_no_inference () {
  DATASET_CHUNK_ID=${1}
  CHUNK_START=${2}
  CHUNK_END=${3}
  export PREPROCESSING_ONLY=1
  export CUDA_VISIBLE_DEVICES=
  python scripts/reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 25 -s "${CHUNK_START}" -e "${CHUNK_END}"
  python scripts/run_pseudo_labelling.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --dataset_name "${PWD}/scripts/reazonspeech_manual_dataloader.py" \
    --dataset_config_name "${DATASET_TYPE}" \
    --dataset_dir_suffix "${CHUNK_START}_${CHUNK_END}" \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 1 \
    --logging_steps 5000 \
    --max_label_length 128 \
    --language "ja" \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
    --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"
  rm -rf "output.${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}"
}

rm -rf "${HOME}/.cache/reazon_manual_download"
for i in {71..80}
do
  TMP_DATA_ID=$(( i + 1 ))
  TMP_CHUNK_START=$(( i * CHUNK_SIZE ))
  TMP_CHUNK_END=$(( TMP_CHUNK_START + CHUNK_SIZE ))
  NEXT_DATA_ID=$(( i + 2 ))
  NEXT_CHUNK_START=$(( (i + 1) * CHUNK_SIZE ))
  NEXT_CHUNK_END=$(( NEXT_CHUNK_START + CHUNK_SIZE ))
  echo "[TMP]  ID: ${TMP_DATA_ID}, START: ${TMP_CHUNK_START}, END: ${TMP_CHUNK_END}"
  echo "[NEXT] ID: ${NEXT_DATA_ID}, START: ${NEXT_CHUNK_START}, END: ${NEXT_CHUNK_END}"
  process_chunk_no_inference ${NEXT_DATA_ID} ${NEXT_CHUNK_START} ${NEXT_CHUNK_END} & process_chunk ${TMP_DATA_ID} ${TMP_CHUNK_START} ${TMP_CHUNK_END}
done
rm -rf "${HOME}/.cache/reazon_manual_download"
process_chunk 82 4050 4095  # remove the last one for eval


for i in 2 32 45 46 55 56 61 62 64 67 68
do
  TMP_DATA_ID=$(( i + 1 ))
  TMP_CHUNK_START=$(( i * CHUNK_SIZE ))
  TMP_CHUNK_END=$(( TMP_CHUNK_START + CHUNK_SIZE ))
  echo "[TMP]  ID: ${TMP_DATA_ID}, START: ${TMP_CHUNK_START}, END: ${TMP_CHUNK_END}"
  process_chunk ${TMP_DATA_ID} ${TMP_CHUNK_START} ${TMP_CHUNK_END}
done

for i in 46 55 56
do
  TMP_DATA_ID=$(( i + 1 ))
  TMP_CHUNK_START=$(( i * CHUNK_SIZE ))
  TMP_CHUNK_END=$(( TMP_CHUNK_START + CHUNK_SIZE ))
  echo "[TMP]  ID: ${TMP_DATA_ID}, START: ${TMP_CHUNK_START}, END: ${TMP_CHUNK_END}"
  process_chunk ${TMP_DATA_ID} ${TMP_CHUNK_START} ${TMP_CHUNK_END}
done


i=61
TMP_DATA_ID=$(( i + 1 ))
TMP_CHUNK_START=$(( i * CHUNK_SIZE ))
TMP_CHUNK_END=$(( TMP_CHUNK_START + CHUNK_SIZE ))
echo "[TMP]  ID: ${TMP_DATA_ID}, START: ${TMP_CHUNK_START}, END: ${TMP_CHUNK_END}"
process_chunk_no_inference ${TMP_DATA_ID} ${TMP_CHUNK_START} ${TMP_CHUNK_END}




CHUNK_END=4095  # remove the last one for eval

#####################
# Filtering Dataset #
#####################
python scripts/run_data_filtering.py \
  -d "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
  --dataset_config_name "${DATASET_TYPE}" \
  --wer_threshold ${WER_THRESHOLD} \
  --text_column_name "transcription" \
  --preprocessing_num_workers 64 \
  --max_label_length 128

############################
# Initialize Student Model #
############################
huggingface-cli repo create "${HF_MODEL_ALIAS}"
git lfs install
git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
cp scripts/create_student_model.py "${HF_MODEL_ALIAS}"
cp scripts/run_distillation.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}"
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"

##########################
# Training Student Model #
##########################
rm -rf run_distillation.py
cp ../scripts/run_distillation.py ./

# Single Step: Log-Mel feature and distillation in a single process
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}" \
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
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 32 \
  --dataloader_num_workers 1 \
  --output_dir "./" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --freeze_encoder \
  --push_to_hub \
  --do_train \
  --overwrite_output_dir \
  --num_train_epochs 8
cd ../


# Two Steps: First, generating Log-Mel feature and save it on HF. Second, distillation where loading the Log-Mel
# feature from HF.
# - Step 1
accelerate launch run_distillation.py \
  --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}" \
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
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 32 \
  --dataloader_num_workers 1 \
  --output_dir "./" \
  --wandb_project "wandb.${HF_MODEL_ALIAS}" \
  --gradient_checkpointing \
  --freeze_encoder \
  --push_to_hub \
  --do_train \
  --overwrite_output_dir \
  --logmel_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized" \
  --num_train_epochs 8

# - Step 2:
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
  --freeze_encoder \
  --push_to_hub \
  --do_train \
  --overwrite_output_dir \
  --skip_logmel_transformation \
  --num_train_epochs 8

##########################
# Evaluate Student Model #
##########################
export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut_basic5000" "asahi417/ja_asr.common_voice_8_0"
do
  accelerate launch scripts/run_short_form_eval.py \
    --model_name_or_path "${HF_ORG}/${HF_MODEL_ALIAS}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size 512 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 1 \
    --generation_max_length 256 \
    --language "ja" \
    --wandb_project "wandb.${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}"
done

#####################################
# (Optional) Evaluate Teacher Model #
#####################################
WHISPER_MODEL="openai/whisper-tiny"
BATCH_SIZE=256
WHISPER_MODEL="openai/whisper-small"
BATCH_SIZE=128
WHISPER_MODEL="openai/whisper-medium"
BATCH_SIZE=64
WHISPER_MODEL="openai/whisper-large-v3"
BATCH_SIZE=32

export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut_basic5000" "asahi417/ja_asr.common_voice_8_0"
do
  accelerate launch run_short_form_eval.py \
    --model_name_or_path "${WHISPER_MODEL}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${WHISPER_MODEL##*/}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --dataloader_num_workers 64 \
    --preprocessing_num_workers 128 \
    --generation_max_length 256 \
    --language "ja" \
    --wandb_project "wandb.${WHISPER_MODEL##*/}.${EVAL_DATASET##*/}"
done


####################
# Trouble Shooting #
####################
# SSL Error
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export CURL_CA_BUNDLE=''

