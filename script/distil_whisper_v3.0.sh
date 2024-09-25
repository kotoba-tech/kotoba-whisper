# kotoba-whisper-v2 distillation version
##########
# Config #
##########
export WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
export TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
export HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts
export HF_MODEL_ALIAS="distil-whisper-bilingual"  # Model alias used when pushing models.
huggingface-cli login  # Configure huggingface.
accelerate config  # Configure accelerate.

################################
# Preprocess Dataset (English) #
################################
process_en_main () {
  DATASET_CONFIG=${1}
  NUM_PROC=${2}
  BATCH=${3}
  accelerate launch --multi_gpu run_pseudo_labelling_v3.py \
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
  accelerate launch --multi_gpu run_pseudo_labelling_v3.py \
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
  python run_data_filtering_v3.py \
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
  python run_data_filtering_v3.py \
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

filter_ja () {
  DATASET_CHUNK_ID=${1}
  python run_data_filtering_v3.py \
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
  python run_data_filtering_v3.py \
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
for i in {0..223}
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
cd "${HF_MODEL_ALIAS}"
python create_student_model.py \
  --teacher_checkpoint "${TEACHER_MODEL}" \
  --encoder_layers 32 \
  --decoder_layers 2 \
  --save_dir "./${HF_MODEL_ALIAS}-init"
cp ./${HF_MODEL_ALIAS}-init/* ./
cd ../
##########################
# Training Student Model #
##########################
```python
import os
from random import shuffle, seed, randint

partion_size = 40
epoch = 8
ja_data_range = list(range(223))
en_data_range = list(range(138))
seed(42)
prev = None
for e in range(1, 1 + epoch):
  print(f"# Epoch {e}")
  shuffle(ja_data_range)
  chunk_size = int(len(ja_data_range)/partion_size)
  ja_config = [ja_data_range[i:i+chunk_size] for i in range(0, len(ja_data_range), chunk_size)][:partion_size]
  shuffle(en_data_range)
  chunk_size = int(len(en_data_range)/partion_size)
  en_config = [en_data_range[i:i+chunk_size] for i in range(0, len(en_data_range), chunk_size)][:partion_size]
  assert len(ja_config) == len(en_config)
  for ja, en in zip(ja_config, en_config):
    random_seed = randint(0, 100000)
    ja = ",".join([f"subset_{x}" for x in ja])
    en = ",".join([f"subset_{x}" for x in en])
    if prev:
      prev_ja, prev_en = prev
      print(f"python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c '{ja}' &")
      print(f"python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c '{en}' &")
      print(f"distillation '{prev_ja}' '{prev_en}' '{random_seed}'")
    prev = [ja, en]
print(f"distillation '{ja}' '{en}' '{random_seed}'")
```


##########################
# Evaluate Student Model #
##########################
# Japanese ASR
for DATA in "japanese-asr/ja_asr.jsut_basic5000" "japanese-asr/ja_asr.reazonspeech_test" "japanese-asr/ja_asr.common_voice_8_0"
do
  python run_short_form_eval.py -l "ja" -t "transcribe" -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "${DATA}" -b 256
done
# English ASR
for DATA_CONFIG in "ami" "earnings22" "tedlium" "librispeech" "voxpopuli"
do
  python run_short_form_eval.py -l "en" -t "transcribe" -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "${DATA_CONFIG}" --dataset-split "validation" --column-text "text" -b 256
done
python run_short_form_eval.py -l "en" -t "transcribe" -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" --dataset-split "validation.clean" --column-text "text" -b 256
# Translation (JA2EN)
for DATA_CONFIG in "fleurs" "covost2"
do
  python run_short_form_eval.py -l "en" -t "translate" -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "japanese-asr/ja2en.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
done
# Translation (EN2JA)
for DATA_CONFIG in "fleurs" "covost2"
do
  python run_short_form_eval.py -l "ja" -t "translate" -m "${HF_ORG}/${HF_MODEL_ALIAS}" -d "japanese-asr/en2ja.s2t_translation" --dataset-config "${DATA_CONFIG}" --dataset-split "test" --column-text "translation" -b 256
done

##########################
# Evaluate Teacher Model #
##########################
for MODEL in "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "openai/whisper-medium" "openai/whisper-small" "openai/whisper-base" "openai/whisper-tiny" "distil-whisper/distil-large-v3"
do
    python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "ami" --dataset-split "validation" --column-text "text" -b 256
    python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "earnings22" --dataset-split "validation" --column-text "text" -b 256
    python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "tedlium" --dataset-split "validation" --column-text "text" -b 256
    python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "librispeech" --dataset-split "validation.clean" --column-text "text" -b 256
    python run_short_form_eval.py -l "en" -t "transcribe" -m "asahi417/distil-whisper-bilingual" -d "japanese-asr/en_asr.esb_eval" --dataset-config "voxpopuli" --dataset-split "validation" --column-text "text" -b 256
done



####################
# Trouble Shooting #
####################
# SSL Error
export REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt'
export CURL_CA_BUNDLE=''


