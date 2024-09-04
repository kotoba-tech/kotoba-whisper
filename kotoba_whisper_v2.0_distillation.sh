# kotoba-whisper-v2 distillation version
##########
# Config #
##########
#WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
ATTN_IMPLEMENTATION="flash_attention_2"  # Use `sdpa` instead if flash attention is not available.
#HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech_mlr.${DATASET_TYPE}"  # Dataset alias used when pushing datasets.
#HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"  # Model alias used when pushing models.
#WARMUP_STEPS=500  # Warmup step.

#huggingface-cli login  # Configure huggingface.
#accelerate config  # Configure accelerate.

################################
# Preprocess Dataset (English) #
################################
process_en_main () {
  DATASET_CONFIG=${1}
  NUM_PROC=${2}
  BATCH=${3}
  accelerate launch --multi_gpu run_pseudo_labelling_v2.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --dataset_name "japanese-asr/en_asr.mls" \
    --dataset_split "train,validation,test" \
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
#  rm -rf "output.en_asr.mls__${DATASET_CONFIG}"
#  rm -rf "${HOME}/.cache/huggingface/datasets/japanese-asr___en_asr.mls/${DATASET_CONFIG}"
}

# runpod_inference
process_en_main "subset_2" 8 64
# runpod_prep_2
process_en_main "subset_0" 8 256
# runpod_prep
process_en_main "subset_9" 8 256
# runpod_ada_1
process_en_main "subset_3" 8 256
# runpod_ada_2
process_en_main "subset_1" 8 256

for i in {0..3}
do
  process_en_main "subset_${i}"
done
process_en_main "subset_${NEXT_DATA_CONFIG}"

################################
# Preprocess Dataset (Japanese) #
################################
generate_en () {
  DATASET_CONFIG=${1}
  accelerate launch --multi_gpu run_pseudo_labelling_v2.py \
    --model_name_or_path "${TEACHER_MODEL}" \
    --dataset_name "japanese-asr/ja_asr.reazon_speech_all" \
    --dataset_split "train" \
    --text_column_name "transcription,transcription/en_gpt3.5" \
    --language "ja,en" \
    --task "transcribe,translate" \
    --dataset_config_name "${DATASET_CONFIG}" \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 1 \
    --logging_steps 5000 \
    --max_label_length 128 \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output.ja_asr.reazon__${DATASET_CONFIG}" \
    --hub_model_id "${HF_ORG}/whisper_transcriptions.reazon_speech_all"
  rm -rf "output.ja_asr.reazon__${DATASET_CONFIG}"
  rm -rf "${HOME}/.cache/huggingface/datasets/japanese-asr___ja_asr.reazon_speech_all/${DATASET_CONFIG}"
}
process_en_pre () {
  export PREPROCESSING_ONLY=1
  export CUDA_VISIBLE_DEVICES=
  generate_en ${1}
}

process_en_main () {
  export PREPROCESSING_ONLY=0
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9  # change it according to the machine
  generate_en ${1}
}

process_en_pre "subset_0"
for i in {0..16}
do
  process_en_pre "subset_$(( i + 1))" & process_en_main "subset_${i}"
done
process_en_main "subset_${NEXT_DATA_CONFIG}"


#####################
# Filtering Dataset #
#####################
DATASET_TYPE="all"  # Dataset type alias.
WARMUP_STEPS=500  # Warmup step.
WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
HF_DATASET_ALIAS="whisper_transcriptions.reazonspeech.${DATASET_TYPE}"  # Dataset alias used when pushing datasets.
HF_MODEL_ALIAS="distil-whisper-large-v3-ja-reazonspeech-${DATASET_TYPE}"  # Model alias used when pushing models.

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


# UNTIL HERE DONE
# FROM HERE TODO


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
  --freeze_encoder \
  --push_to_hub \
  --do_train \
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


