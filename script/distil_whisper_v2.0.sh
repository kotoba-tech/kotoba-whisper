# kotoba-whisper-v1.2 is whisper distillation with ReazonSpeech Large subset. This model is still in progress.
##########
# Config #
##########
DATASET_TYPE="all"  # Dataset type alias.
WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"  # Dataset alias used when pushing datasets.
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"  # Model alias used when pushing models.
WARMUP_STEPS=500  # Warmup step.
huggingface-cli login  # Configure huggingface.
accelerate config  # Configure accelerate.

######################
# Preprocess Dataset #
######################
CHUNK_SIZE=50
process_chunk () {
  DATASET_CHUNK_ID=${1}
  CHUNK_START=${2}
  CHUNK_END=${3}
  export PREPROCESSING_ONLY=0
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9  # change it according to the machine
  python reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 25 -s "${CHUNK_START}" -e "${CHUNK_END}"
  accelerate launch --multi_gpu run_pseudo_labelling.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --attn_implementation "sdpa" \
    --dataset_name "${PWD}/reazonspeech_manual_dataloader.py" \
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
  python reazonspeech_manual_downloader.py -t "${DATASET_TYPE}" -p 25 -s "${CHUNK_START}" -e "${CHUNK_END}"
  python run_pseudo_labelling.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --attn_implementation "sdpa" \
    --dataset_name "${PWD}/reazonspeech_manual_dataloader.py" \
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

for i in {1..80}
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
process_chunk 82 4050 4095  # remove the last one for eval

#####################
# Filtering Dataset #
#####################
for DATASET_CHUNK_ID in {1..82}
do
  python run_data_filtering.py \
    -d "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}" \
    --dataset_config_name "${DATASET_TYPE}" \
    --wer_threshold ${WER_THRESHOLD} \
    --text_column_name "transcription" \
    --preprocessing_num_workers 1 \
    --preprocessing_batch_size 64 \
    --max_label_length 128 \
    --skip_logmel
  rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___whisper_transcriptions.reazonspeech.all_${DATASET_CHUNK_ID}*"
done

for DATASET_CHUNK_ID in {1..82}
do
  python run_data_filtering.py \
    -d "${HF_ORG}/${HF_DATASET_ALIAS}_${DATASET_CHUNK_ID}.wer_${WER_THRESHOLD}" \
    --dataset_config_name "${DATASET_TYPE}" \
    --wer_threshold ${WER_THRESHOLD} \
    --text_column_name "transcription" \
    --preprocessing_num_workers 1 \
    --preprocessing_batch_size 64 \
    --max_label_length 128 \
    --skip_filtering
  rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___whisper_transcriptions.reazonspeech.all_${DATASET_CHUNK_ID}.wer_${WER_THRESHOLD}"
done

#################
# Merge Dataset #
#################
python misc/merge_reazon_all_dataset.py

############################
# Initialize Student Model #
############################
huggingface-cli repo create "${HF_MODEL_ALIAS}"
git lfs install
git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
cp create_student_model.py "${HF_MODEL_ALIAS}"
cd "${HF_MODEL_ALIAS}"
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"
git add . && git commit -m "init" && git push
cd ../

##########################
# Training Student Model #
##########################
distillation () {
  export WANDB_DISABLED="true"
  MODEL_NAME=${1}
  MODEL_CONFIG=${2}
  WARMUP_STEPS=${3}
  SEED=${4}
  echo "MODEL_NAME  : ${MODEL_NAME}"
  echo "MODEL_CONFIG: ${MODEL_CONFIG}"
  echo "WARMUP_STEPS: ${WARMUP_STEPS}"
  accelerate launch run_distillation.py \
    --model_name_or_path "${MODEL_NAME}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --train_dataset_name "${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized" \
    --train_dataset_config_name "${MODEL_CONFIG}" \
    --language "ja" \
    --attn_implementation "flash_attention_2" \
    --max_label_length 128 \
    --train_split_name "train" \
    --save_steps 5000 \
    --warmup_steps "${WARMUP_STEPS}" \
    --learning_rate 0.0001 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 100 \
    --save_total_limit 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 1 \
    --output_dir "./${HF_MODEL_ALIAS}" \
    --wandb_project "wandb.${HF_MODEL_ALIAS}" \
    --gradient_checkpointing \
    --freeze_encoder \
    --do_train \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --seed ${SEED} \
    --push_to_hub
}

git clone "https://huggingface.co/${HF_ORG}/${HF_MODEL_ALIAS}"
python -c """from datasets import load_dataset; load_dataset('${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized', 'split_0', num_proc=16)"""
for i in {1..8}
do
  echo "EPOCH ${i}"
  for s in {0..8}
  do
    if [ ${s} = 8 ]; then
      python -c """from datasets import load_dataset; load_dataset('${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized', 'split_0', num_proc=16)""" &
    else
      python -c """from datasets import load_dataset; load_dataset('${HF_ORG}/${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized', 'split_$(( s + 1 ))', num_proc=16)""" &
    fi
    rm -rf ${HF_MODEL_ALIAS}/checkpoint-*
    if [ ${i} -eq 1 ] && [ ${s} = 0 ]; then
      distillation "${HF_MODEL_ALIAS}/${HF_MODEL_ALIAS}-init" "split_${s}" "${WARMUP_STEPS}" ${i}
      rm -rf "${HF_MODEL_ALIAS}/${HF_MODEL_ALIAS}-init"
    else
      distillation "${HF_MODEL_ALIAS}" "split_${s}" "0" ${i}
    fi
    rm -rf "${HOME}/.cache/huggingface/datasets/${HF_ORG}___${HF_DATASET_ALIAS}.wer_${WER_THRESHOLD}.vectorized/split_${s}"
    rm -rf "${HOME}/.cache/huggingface/datasets/downloads"
  done
done

##########################
# Evaluate Student Model #
##########################
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
    python run_short_form_eval.py -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "${DATA}" -b 128
done

####################
# Trouble Shooting #
####################
# SSL Error
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export CURL_CA_BUNDLE=''


