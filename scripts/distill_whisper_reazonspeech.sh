####################
# Select Data Size #
####################
DATASET_TYPE="tiny"
#DATASET_TYPE="small"
#DATASET_TYPE="medium"
#DATASET_TYPE="large"

##########
# Config #
##########
WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"  # Dataset alias used when pushing datasets.
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"  # Model alias used when pushing models.
# Warmup step.
if [ "${DATASET_TYPE}" = "tiny" ]; then
  WARMUP_STEPS=10
elif [ "${DATASET_TYPE}" = "small" ]; then
  WARMUP_STEPS=25
elif [ "${DATASET_TYPE}" = "medium" ]; then
  WARMUP_STEPS=50
else
  WARMUP_STEPS=100
fi
huggingface-cli login  # Configure huggingface.

####################
# Download Dataset #
####################
python ./reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 100

###################
# Generate Labels #
###################
export WANDB_DISABLED="true"
accelerate launch --multi_gpu ./run_pseudo_labelling.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${PWD}/reazonspeech_manual_dataloader.py" \
  --dataset_config_name "${DATASET_TYPE}" \
  --dataset_split_name "train" \
  --text_column_name "transcription" \
  --id_column_name "name" \
  --per_device_eval_batch_size 4 \
  --dataloader_num_workers 32 \
  --preprocessing_num_workers 32 \
  --logging_steps 100 \
  --max_label_length 128 \
  --language "ja" \
  --return_timestamps \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --overwrite_output_dir \
  --output_dir "output.${HF_DATASET_ALIAS}" \
  --wandb_project "wandb.${HF_DATASET_ALIAS}" \
  --hub_model_id "${HF_ORG}/${HF_DATASET_ALIAS}" \
  --push_to_hub

#####################
# Filtering Dataset #
#####################
python ./run_data_filtering.py \
  -d "${HF_ORG}/${HF_DATASET_ALIAS}" \
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
cp ./create_student_model.py "${HF_MODEL_ALIAS}"
cp ./run_distillation.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}"
python create_student_model.py --teacher_checkpoint "${TEACHER_MODEL}" \
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
  --freeze_encoder \
  --push_to_hub \
  --do_train \
  --overwrite_output_dir \
  --num_train_epochs 8

##########################
# Evaluate Student Model #
##########################
export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut_basic5000" "asahi417/ja_asr.common_voice_8_0" "asahi417/ja_asr.reazonspeech_test"
do
  accelerate launch ./run_short_form_eval.py \
    --model_name_or_path "${HF_ORG}/${HF_MODEL_ALIAS}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${HF_MODEL_ALIAS}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size 4 \
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
WHISPER_MODEL="openai/whisper-small"
WHISPER_MODEL="openai/whisper-medium"
WHISPER_MODEL="openai/whisper-large-v3"

if [ "${WHISPER_MODEL}" = "openai/whisper-tiny" ]; then
  BATCH_SIZE=256
elif [ "${WHISPER_MODEL}" = "openai/whisper-small" ]; then
  BATCH_SIZE=128
elif [ "${WHISPER_MODEL}" = "openai/whisper-medium" ]; then
  BATCH_SIZE=64
else
  BATCH_SIZE=32
fi
export WANDB_DISABLED="true"
for EVAL_DATASET in "asahi417/ja_asr.jsut_basic5000" "asahi417/ja_asr.common_voice_8_0" "asahi417/ja_asr.reazonspeech_test"
do
  accelerate launch ./run_short_form_eval.py \
    --model_name_or_path "${WHISPER_MODEL}" \
    --dataset_name "${EVAL_DATASET}" \
    --dataset_split_name "test" \
    --text_column_name "transcription" \
    --output_dir "eval/${WHISPER_MODEL##*/}.${EVAL_DATASET##*/}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --dataloader_num_workers 32 \
    --preprocessing_num_workers 32 \
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

