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
HF_ORG="asahi417"
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
distillation () {
  export WANDB_DISABLED="true"
  MODEL_CONFIG_1=${1}
  MODEL_CONFIG_2=${2}
  SEED=${3}
  echo "MODEL_CONFIG_1: ${MODEL_CONFIG_1}"
  echo "MODEL_CONFIG_2: ${MODEL_CONFIG_2}"
  echo "SEED: ${SEED}"
  accelerate launch run_distillation_v3.py \
    --model_name_or_path "${HF_MODEL_ALIAS}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --dataset_name_1 "${HF_ORG}/whisper_transcriptions.reazon_speech_all.wer_${WER_THRESHOLD}.vectorized" \
    --dataset_split_name_1 "train" \
    --dataset_config_name_1 "${MODEL_CONFIG_1}" \
    --dataset_feature_1 "whisper_transcription,transcription/en_gpt3.5" \
    --dataset_language_1 "ja,en" \
    --dataset_task_1 "transcribe,translate" \
    --dataset_timestamp_1 "true,false" \
    --dataset_kl_1 "true,false" \
    --dataset_name_2 "${HF_ORG}/whisper_transcriptions.mls.wer_${WER_THRESHOLD}.vectorized" \
    --dataset_split_name_2 "train" \
    --dataset_config_name_2 "${MODEL_CONFIG_1}" \
    --dataset_feature_2 "whisper_transcription,transcription/ja_gpt3.5" \
    --dataset_language_2 "en,ja" \
    --dataset_task_2 "transcribe,translate" \
    --dataset_timestamp_2 "true,false" \
    --dataset_kl_2 "true,false" \
    --max_label_length 128 \
    --learning_rate 0.0001 \
    --logging_steps 50 \
    --attn_implementation "flash_attention_2" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_workers 16 \
    --dataloader_num_workers 1 \
    --output_dir "./${HF_MODEL_ALIAS}" \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --seed ${SEED} \
    --report_to "none" \
    --num_train_epochs 1
}
```python
import os
from random import shuffle, seed, randint

partion_size = 15
epoch = 8
ja_data_range = list(range(223))
en_data_range = list(range(138))
seed(42)
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
    ja = ",".join([f"subset_{x}" for x in ja])
    en = ",".join([f"subset_{x}" for x in en])
    random_seed = randint(0, 100000)
    print(f"distillation '{ja}' '{en}' '{random_seed}'")
```

# Epoch 1
distillation 'subset_162,subset_36,subset_185,subset_2,subset_45,subset_105,subset_195,subset_168,subset_52,subset_208,subset_212,subset_159,subset_110,subset_123' 'subset_67,subset_121,subset_126,subset_78,subset_134,subset_14,subset_72,subset_113,subset_4' '17154'
distillation 'subset_121,subset_98,subset_181,subset_60,subset_13,subset_44,subset_16,subset_84,subset_135,subset_29,subset_215,subset_21,subset_169,subset_194' 'subset_26,subset_131,subset_38,subset_74,subset_40,subset_75,subset_98,subset_117,subset_22' '88039'
distillation 'subset_30,subset_127,subset_175,subset_4,subset_42,subset_47,subset_5,subset_77,subset_3,subset_15,subset_216,subset_190,subset_205,subset_81' 'subset_53,subset_47,subset_48,subset_64,subset_63,subset_25,subset_19,subset_28,subset_5' '84607'
distillation 'subset_192,subset_99,subset_133,subset_66,subset_145,subset_73,subset_10,subset_122,subset_177,subset_9,subset_12,subset_38,subset_138,subset_61' 'subset_107,subset_2,subset_81,subset_65,subset_36,subset_10,subset_33,subset_32,subset_127' '39321'
distillation 'subset_198,subset_89,subset_193,subset_134,subset_119,subset_79,subset_78,subset_209,subset_64,subset_103,subset_0,subset_111,subset_196,subset_180' 'subset_41,subset_39,subset_20,subset_49,subset_34,subset_37,subset_3,subset_91,subset_106' '59929'
distillation 'subset_37,subset_222,subset_140,subset_126,subset_104,subset_144,subset_187,subset_218,subset_32,subset_171,subset_152,subset_85,subset_48,subset_161' 'subset_76,subset_79,subset_80,subset_105,subset_42,subset_97,subset_99,subset_46,subset_44' '41441'
distillation 'subset_125,subset_76,subset_100,subset_153,subset_128,subset_19,subset_131,subset_199,subset_156,subset_115,subset_65,subset_184,subset_221,subset_46' 'subset_66,subset_129,subset_29,subset_133,subset_50,subset_122,subset_87,subset_0,subset_88' '98548'
distillation 'subset_51,subset_101,subset_54,subset_172,subset_106,subset_109,subset_142,subset_149,subset_95,subset_167,subset_157,subset_33,subset_164,subset_130' 'subset_77,subset_95,subset_94,subset_124,subset_61,subset_125,subset_120,subset_21,subset_30' '9508'
distillation 'subset_82,subset_113,subset_197,subset_63,subset_132,subset_124,subset_186,subset_174,subset_112,subset_72,subset_120,subset_148,subset_27,subset_201' 'subset_11,subset_136,subset_123,subset_114,subset_96,subset_70,subset_137,subset_9,subset_90' '1220'
distillation 'subset_34,subset_102,subset_80,subset_206,subset_155,subset_14,subset_83,subset_191,subset_69,subset_211,subset_118,subset_146,subset_217,subset_136' 'subset_102,subset_35,subset_23,subset_104,subset_132,subset_57,subset_68,subset_89,subset_110' '60068'
distillation 'subset_43,subset_18,subset_68,subset_53,subset_90,subset_94,subset_41,subset_93,subset_116,subset_182,subset_176,subset_25,subset_202,subset_165' 'subset_115,subset_13,subset_43,subset_101,subset_51,subset_7,subset_108,subset_82,subset_83' '81416'
distillation 'subset_74,subset_58,subset_170,subset_17,subset_49,subset_147,subset_92,subset_158,subset_160,subset_75,subset_141,subset_20,subset_96,subset_31' 'subset_86,subset_6,subset_93,subset_59,subset_111,subset_54,subset_45,subset_55,subset_130' '73792'
distillation 'subset_137,subset_117,subset_11,subset_67,subset_200,subset_88,subset_91,subset_24,subset_97,subset_204,subset_213,subset_86,subset_203,subset_39' 'subset_109,subset_12,subset_24,subset_52,subset_103,subset_116,subset_100,subset_31,subset_128' '13104'
distillation 'subset_214,subset_87,subset_207,subset_178,subset_40,subset_1,subset_71,subset_150,subset_114,subset_56,subset_107,subset_210,subset_179,subset_166' 'subset_118,subset_73,subset_112,subset_119,subset_92,subset_16,subset_69,subset_27,subset_62' '9602'
distillation 'subset_183,subset_50,subset_143,subset_220,subset_154,subset_129,subset_59,subset_55,subset_23,subset_7,subset_8,subset_108,subset_151,subset_22' 'subset_85,subset_71,subset_60,subset_135,subset_84,subset_8,subset_17,subset_58,subset_15' '70468'
# Epoch 2
distillation 'subset_36,subset_57,subset_94,subset_4,subset_177,subset_10,subset_128,subset_165,subset_69,subset_73,subset_147,subset_62,subset_175,subset_58' 'subset_91,subset_136,subset_92,subset_97,subset_10,subset_38,subset_71,subset_2,subset_68' '98176'
distillation 'subset_151,subset_70,subset_168,subset_41,subset_93,subset_143,subset_17,subset_97,subset_134,subset_68,subset_204,subset_86,subset_14,subset_181' 'subset_56,subset_101,subset_78,subset_117,subset_1,subset_25,subset_130,subset_14,subset_35' '70702'
distillation 'subset_7,subset_84,subset_171,subset_96,subset_173,subset_137,subset_80,subset_40,subset_178,subset_217,subset_179,subset_109,subset_163,subset_186' 'subset_100,subset_40,subset_126,subset_79,subset_125,subset_110,subset_53,subset_74,subset_99' '6826'
distillation 'subset_3,subset_132,subset_88,subset_99,subset_102,subset_79,subset_198,subset_110,subset_222,subset_106,subset_13,subset_27,subset_56,subset_16' 'subset_49,subset_0,subset_134,subset_6,subset_70,subset_13,subset_28,subset_64,subset_80' '45870'
distillation 'subset_154,subset_152,subset_148,subset_144,subset_92,subset_89,subset_91,subset_98,subset_161,subset_184,subset_112,subset_85,subset_145,subset_72' 'subset_119,subset_37,subset_123,subset_9,subset_16,subset_82,subset_75,subset_20,subset_60' '29389'
distillation 'subset_221,subset_0,subset_200,subset_54,subset_105,subset_25,subset_207,subset_126,subset_122,subset_116,subset_101,subset_22,subset_191,subset_39' 'subset_104,subset_94,subset_55,subset_129,subset_114,subset_63,subset_67,subset_128,subset_19' '85215'
distillation 'subset_197,subset_61,subset_129,subset_213,subset_139,subset_156,subset_104,subset_135,subset_55,subset_121,subset_202,subset_95,subset_155,subset_100' 'subset_57,subset_105,subset_89,subset_45,subset_17,subset_106,subset_58,subset_77,subset_86' '8993'
distillation 'subset_167,subset_103,subset_48,subset_174,subset_63,subset_51,subset_52,subset_77,subset_192,subset_9,subset_215,subset_125,subset_2,subset_120' 'subset_116,subset_31,subset_5,subset_113,subset_11,subset_69,subset_93,subset_23,subset_46' '85426'
distillation 'subset_183,subset_34,subset_172,subset_83,subset_43,subset_133,subset_193,subset_124,subset_150,subset_21,subset_158,subset_49,subset_141,subset_45' 'subset_44,subset_4,subset_8,subset_12,subset_42,subset_81,subset_47,subset_112,subset_131' '5276'
distillation 'subset_130,subset_46,subset_166,subset_26,subset_196,subset_90,subset_118,subset_67,subset_146,subset_195,subset_149,subset_66,subset_108,subset_119' 'subset_120,subset_103,subset_3,subset_32,subset_115,subset_41,subset_26,subset_24,subset_51' '98858'
distillation 'subset_107,subset_142,subset_131,subset_220,subset_209,subset_71,subset_199,subset_176,subset_115,subset_32,subset_212,subset_42,subset_164,subset_15' 'subset_65,subset_90,subset_15,subset_124,subset_127,subset_102,subset_27,subset_52,subset_33' '4067'
distillation 'subset_20,subset_182,subset_35,subset_208,subset_219,subset_216,subset_44,subset_30,subset_23,subset_53,subset_33,subset_210,subset_113,subset_81' 'subset_111,subset_87,subset_7,subset_48,subset_133,subset_21,subset_66,subset_95,subset_43' '32411'
distillation 'subset_214,subset_75,subset_189,subset_76,subset_162,subset_159,subset_37,subset_157,subset_1,subset_29,subset_123,subset_64,subset_201,subset_28' 'subset_54,subset_137,subset_135,subset_108,subset_34,subset_96,subset_62,subset_61,subset_132' '26130'
distillation 'subset_50,subset_160,subset_24,subset_12,subset_153,subset_87,subset_38,subset_74,subset_188,subset_180,subset_190,subset_59,subset_114,subset_194' 'subset_122,subset_18,subset_22,subset_50,subset_98,subset_84,subset_118,subset_36,subset_30' '2671'
distillation 'subset_127,subset_6,subset_5,subset_169,subset_117,subset_187,subset_18,subset_11,subset_185,subset_211,subset_31,subset_8,subset_170,subset_218' 'subset_72,subset_73,subset_76,subset_29,subset_109,subset_107,subset_121,subset_39,subset_88' '81439'
# Epoch 3
distillation 'subset_172,subset_183,subset_38,subset_30,subset_24,subset_17,subset_158,subset_31,subset_43,subset_114,subset_208,subset_121,subset_215,subset_110' 'subset_108,subset_55,subset_128,subset_84,subset_15,subset_117,subset_52,subset_1,subset_129' '15396'
distillation 'subset_108,subset_35,subset_12,subset_137,subset_22,subset_151,subset_54,subset_194,subset_18,subset_6,subset_207,subset_81,subset_58,subset_200' 'subset_63,subset_114,subset_49,subset_19,subset_104,subset_127,subset_37,subset_5,subset_43' '84896'
distillation 'subset_162,subset_34,subset_154,subset_98,subset_188,subset_146,subset_73,subset_132,subset_113,subset_27,subset_86,subset_177,subset_107,subset_63' 'subset_10,subset_35,subset_107,subset_30,subset_136,subset_96,subset_73,subset_111,subset_53' '20181'
distillation 'subset_82,subset_122,subset_164,subset_46,subset_50,subset_47,subset_150,subset_144,subset_51,subset_36,subset_5,subset_192,subset_166,subset_65' 'subset_137,subset_3,subset_83,subset_62,subset_112,subset_65,subset_20,subset_33,subset_85' '65323'
distillation 'subset_52,subset_69,subset_167,subset_93,subset_161,subset_214,subset_71,subset_10,subset_190,subset_168,subset_57,subset_179,subset_136,subset_8' 'subset_47,subset_94,subset_59,subset_22,subset_14,subset_11,subset_77,subset_118,subset_121' '93926'
distillation 'subset_118,subset_33,subset_101,subset_175,subset_178,subset_103,subset_25,subset_40,subset_206,subset_189,subset_61,subset_68,subset_212,subset_142' 'subset_69,subset_88,subset_124,subset_95,subset_70,subset_60,subset_17,subset_45,subset_122' '38254'
distillation 'subset_163,subset_204,subset_70,subset_185,subset_111,subset_77,subset_80,subset_148,subset_156,subset_219,subset_218,subset_48,subset_95,subset_195' 'subset_71,subset_75,subset_126,subset_133,subset_99,subset_23,subset_6,subset_46,subset_41' '66698'
distillation 'subset_88,subset_85,subset_196,subset_220,subset_79,subset_105,subset_197,subset_87,subset_96,subset_28,subset_106,subset_7,subset_217,subset_62' 'subset_86,subset_120,subset_110,subset_80,subset_115,subset_113,subset_58,subset_125,subset_135' '92459'
distillation 'subset_19,subset_59,subset_152,subset_20,subset_135,subset_32,subset_92,subset_29,subset_119,subset_131,subset_90,subset_205,subset_13,subset_53' 'subset_90,subset_92,subset_29,subset_36,subset_78,subset_97,subset_13,subset_81,subset_82' '35838'
distillation 'subset_201,subset_97,subset_198,subset_94,subset_221,subset_199,subset_37,subset_112,subset_104,subset_56,subset_141,subset_102,subset_39,subset_130' 'subset_105,subset_38,subset_98,subset_101,subset_27,subset_116,subset_34,subset_31,subset_54' '54459'
distillation 'subset_165,subset_129,subset_100,subset_193,subset_11,subset_83,subset_0,subset_134,subset_153,subset_191,subset_145,subset_74,subset_216,subset_140' 'subset_67,subset_51,subset_123,subset_24,subset_109,subset_130,subset_89,subset_119,subset_76' '63242'
distillation 'subset_49,subset_66,subset_72,subset_67,subset_99,subset_120,subset_138,subset_133,subset_23,subset_55,subset_2,subset_117,subset_45,subset_9' 'subset_102,subset_42,subset_9,subset_39,subset_79,subset_2,subset_72,subset_100,subset_66' '61894'
distillation 'subset_4,subset_213,subset_210,subset_26,subset_41,subset_203,subset_75,subset_125,subset_149,subset_139,subset_147,subset_78,subset_171,subset_15' 'subset_64,subset_40,subset_103,subset_0,subset_93,subset_25,subset_57,subset_74,subset_48' '31946'
distillation 'subset_157,subset_126,subset_123,subset_14,subset_91,subset_44,subset_1,subset_42,subset_143,subset_222,subset_211,subset_174,subset_155,subset_159' 'subset_32,subset_12,subset_131,subset_7,subset_87,subset_28,subset_18,subset_134,subset_26' '59871'
distillation 'subset_176,subset_187,subset_128,subset_115,subset_181,subset_116,subset_186,subset_169,subset_60,subset_160,subset_180,subset_170,subset_182,subset_3' 'subset_56,subset_132,subset_44,subset_4,subset_106,subset_68,subset_8,subset_50,subset_61' '72255'
# Epoch 4
distillation 'subset_200,subset_207,subset_23,subset_32,subset_184,subset_190,subset_57,subset_134,subset_192,subset_80,subset_199,subset_180,subset_97,subset_181' 'subset_47,subset_26,subset_34,subset_118,subset_82,subset_29,subset_4,subset_110,subset_9' '36046'
distillation 'subset_170,subset_218,subset_209,subset_112,subset_24,subset_90,subset_71,subset_133,subset_20,subset_85,subset_156,subset_1,subset_35,subset_64' 'subset_96,subset_57,subset_114,subset_80,subset_92,subset_104,subset_121,subset_30,subset_109' '81830'
distillation 'subset_146,subset_191,subset_168,subset_11,subset_47,subset_125,subset_193,subset_37,subset_58,subset_219,subset_127,subset_17,subset_36,subset_92' 'subset_111,subset_67,subset_65,subset_11,subset_79,subset_46,subset_135,subset_21,subset_6' '82978'
distillation 'subset_91,subset_41,subset_162,subset_67,subset_33,subset_63,subset_124,subset_159,subset_61,subset_43,subset_69,subset_56,subset_217,subset_50' 'subset_134,subset_22,subset_14,subset_81,subset_3,subset_40,subset_72,subset_102,subset_20' '76724'
distillation 'subset_40,subset_122,subset_12,subset_99,subset_215,subset_107,subset_76,subset_77,subset_15,subset_143,subset_106,subset_158,subset_149,subset_132' 'subset_115,subset_116,subset_84,subset_39,subset_119,subset_95,subset_107,subset_94,subset_76' '71874'
distillation 'subset_51,subset_38,subset_28,subset_109,subset_55,subset_46,subset_100,subset_117,subset_7,subset_110,subset_60,subset_144,subset_176,subset_155' 'subset_136,subset_91,subset_75,subset_55,subset_120,subset_101,subset_78,subset_17,subset_125' '93343'
distillation 'subset_29,subset_196,subset_166,subset_105,subset_203,subset_182,subset_164,subset_52,subset_83,subset_216,subset_8,subset_49,subset_138,subset_120' 'subset_33,subset_48,subset_63,subset_71,subset_74,subset_108,subset_28,subset_133,subset_105' '42622'
distillation 'subset_183,subset_174,subset_79,subset_211,subset_210,subset_3,subset_157,subset_34,subset_115,subset_178,subset_10,subset_154,subset_72,subset_189' 'subset_10,subset_90,subset_132,subset_100,subset_0,subset_129,subset_53,subset_68,subset_112' '49925'
distillation 'subset_44,subset_84,subset_206,subset_208,subset_18,subset_194,subset_165,subset_21,subset_202,subset_160,subset_5,subset_66,subset_98,subset_22' 'subset_77,subset_61,subset_128,subset_123,subset_49,subset_32,subset_37,subset_73,subset_89' '78301'
distillation 'subset_65,subset_48,subset_148,subset_103,subset_108,subset_142,subset_171,subset_31,subset_169,subset_82,subset_59,subset_123,subset_2,subset_62' 'subset_60,subset_1,subset_69,subset_38,subset_64,subset_44,subset_42,subset_88,subset_106' '69541'
distillation 'subset_6,subset_167,subset_175,subset_161,subset_16,subset_26,subset_39,subset_13,subset_78,subset_139,subset_141,subset_9,subset_54,subset_220' 'subset_51,subset_8,subset_16,subset_93,subset_31,subset_18,subset_85,subset_45,subset_13' '38652'
distillation 'subset_70,subset_179,subset_119,subset_151,subset_177,subset_197,subset_113,subset_30,subset_42,subset_104,subset_75,subset_19,subset_81,subset_163' 'subset_41,subset_131,subset_50,subset_66,subset_35,subset_24,subset_103,subset_36,subset_122' '59469'
distillation 'subset_74,subset_198,subset_188,subset_114,subset_201,subset_128,subset_45,subset_213,subset_126,subset_195,subset_131,subset_68,subset_121,subset_87' 'subset_58,subset_54,subset_130,subset_27,subset_124,subset_99,subset_70,subset_56,subset_113' '66263'
distillation 'subset_93,subset_89,subset_0,subset_214,subset_147,subset_173,subset_212,subset_135,subset_95,subset_130,subset_129,subset_204,subset_111,subset_205' 'subset_25,subset_62,subset_19,subset_127,subset_86,subset_59,subset_15,subset_87,subset_117' '79354'
distillation 'subset_102,subset_152,subset_86,subset_53,subset_4,subset_153,subset_145,subset_25,subset_14,subset_101,subset_172,subset_136,subset_94,subset_116' 'subset_43,subset_83,subset_2,subset_97,subset_12,subset_5,subset_126,subset_98,subset_52' '56387'
# Epoch 5
distillation 'subset_192,subset_179,subset_180,subset_14,subset_214,subset_6,subset_195,subset_91,subset_106,subset_131,subset_78,subset_201,subset_95,subset_142' 'subset_46,subset_131,subset_81,subset_101,subset_26,subset_77,subset_15,subset_2,subset_133' '94526'
distillation 'subset_48,subset_211,subset_176,subset_108,subset_32,subset_168,subset_7,subset_155,subset_173,subset_42,subset_99,subset_55,subset_80,subset_74' 'subset_22,subset_64,subset_73,subset_104,subset_10,subset_102,subset_118,subset_6,subset_34' '57366'
distillation 'subset_97,subset_116,subset_203,subset_41,subset_37,subset_204,subset_44,subset_43,subset_19,subset_169,subset_215,subset_52,subset_212,subset_2' 'subset_119,subset_99,subset_42,subset_112,subset_121,subset_13,subset_24,subset_40,subset_55' '5726'
distillation 'subset_3,subset_159,subset_144,subset_24,subset_165,subset_148,subset_51,subset_107,subset_69,subset_45,subset_216,subset_185,subset_93,subset_183' 'subset_14,subset_47,subset_130,subset_74,subset_134,subset_113,subset_72,subset_19,subset_53' '53852'
distillation 'subset_118,subset_157,subset_23,subset_63,subset_26,subset_27,subset_184,subset_67,subset_15,subset_119,subset_124,subset_209,subset_20,subset_88' 'subset_51,subset_105,subset_44,subset_78,subset_8,subset_132,subset_21,subset_41,subset_76' '47749'
distillation 'subset_77,subset_53,subset_89,subset_98,subset_219,subset_47,subset_123,subset_79,subset_127,subset_222,subset_200,subset_82,subset_5,subset_61' 'subset_16,subset_88,subset_60,subset_125,subset_48,subset_116,subset_117,subset_11,subset_58' '88652'
distillation 'subset_146,subset_190,subset_65,subset_196,subset_158,subset_126,subset_46,subset_22,subset_189,subset_59,subset_120,subset_31,subset_162,subset_170' 'subset_87,subset_122,subset_33,subset_18,subset_30,subset_111,subset_93,subset_129,subset_59' '94317'
distillation 'subset_152,subset_206,subset_113,subset_54,subset_56,subset_128,subset_117,subset_76,subset_75,subset_150,subset_40,subset_193,subset_21,subset_9' 'subset_80,subset_86,subset_9,subset_107,subset_4,subset_38,subset_17,subset_66,subset_54' '31133'
distillation 'subset_194,subset_16,subset_171,subset_71,subset_90,subset_139,subset_84,subset_199,subset_100,subset_28,subset_202,subset_174,subset_92,subset_83' 'subset_27,subset_91,subset_65,subset_100,subset_0,subset_103,subset_7,subset_3,subset_45' '58222'
distillation 'subset_104,subset_151,subset_36,subset_105,subset_172,subset_178,subset_138,subset_186,subset_135,subset_73,subset_62,subset_129,subset_101,subset_58' 'subset_79,subset_106,subset_57,subset_94,subset_92,subset_85,subset_97,subset_137,subset_35' '80031'
distillation 'subset_70,subset_207,subset_197,subset_136,subset_198,subset_132,subset_33,subset_134,subset_149,subset_57,subset_177,subset_121,subset_4,subset_30' 'subset_28,subset_32,subset_95,subset_135,subset_69,subset_61,subset_98,subset_29,subset_31' '37346'
distillation 'subset_39,subset_38,subset_140,subset_8,subset_187,subset_182,subset_64,subset_122,subset_66,subset_205,subset_217,subset_103,subset_13,subset_35' 'subset_68,subset_128,subset_56,subset_120,subset_52,subset_49,subset_37,subset_114,subset_126' '98617'
distillation 'subset_111,subset_110,subset_220,subset_109,subset_181,subset_10,subset_86,subset_164,subset_94,subset_11,subset_114,subset_29,subset_87,subset_161' 'subset_25,subset_115,subset_36,subset_75,subset_83,subset_82,subset_71,subset_108,subset_39' '98112'
distillation 'subset_141,subset_218,subset_175,subset_125,subset_137,subset_145,subset_96,subset_130,subset_133,subset_85,subset_112,subset_160,subset_213,subset_17' 'subset_96,subset_90,subset_23,subset_110,subset_123,subset_20,subset_50,subset_12,subset_5' '58841'
distillation 'subset_49,subset_143,subset_188,subset_153,subset_154,subset_60,subset_156,subset_147,subset_115,subset_210,subset_18,subset_166,subset_34,subset_12' 'subset_63,subset_127,subset_109,subset_124,subset_84,subset_89,subset_1,subset_70,subset_62' '30671'
# Epoch 6
distillation 'subset_116,subset_123,subset_167,subset_211,subset_149,subset_124,subset_22,subset_28,subset_44,subset_50,subset_104,subset_114,subset_11,subset_94' 'subset_104,subset_46,subset_21,subset_62,subset_129,subset_36,subset_100,subset_24,subset_82' '3718'
distillation 'subset_81,subset_67,subset_30,subset_20,subset_16,subset_208,subset_140,subset_46,subset_13,subset_168,subset_48,subset_112,subset_182,subset_206' 'subset_112,subset_124,subset_10,subset_131,subset_119,subset_20,subset_58,subset_30,subset_125' '10646'
distillation 'subset_1,subset_184,subset_93,subset_4,subset_47,subset_61,subset_105,subset_202,subset_68,subset_189,subset_60,subset_204,subset_29,subset_158' 'subset_114,subset_15,subset_117,subset_81,subset_56,subset_86,subset_3,subset_19,subset_25' '5958'
distillation 'subset_134,subset_201,subset_216,subset_99,subset_135,subset_132,subset_74,subset_83,subset_103,subset_160,subset_145,subset_186,subset_14,subset_56' 'subset_64,subset_94,subset_77,subset_71,subset_32,subset_2,subset_38,subset_136,subset_52' '84076'
distillation 'subset_57,subset_128,subset_32,subset_58,subset_137,subset_51,subset_110,subset_21,subset_76,subset_69,subset_177,subset_52,subset_212,subset_53' 'subset_85,subset_79,subset_45,subset_109,subset_87,subset_103,subset_98,subset_41,subset_130' '75598'
distillation 'subset_96,subset_65,subset_205,subset_122,subset_133,subset_23,subset_162,subset_196,subset_38,subset_172,subset_63,subset_173,subset_5,subset_159' 'subset_65,subset_90,subset_132,subset_89,subset_39,subset_61,subset_12,subset_11,subset_122' '34688'
distillation 'subset_72,subset_18,subset_117,subset_108,subset_3,subset_181,subset_86,subset_200,subset_146,subset_59,subset_17,subset_144,subset_221,subset_92' 'subset_126,subset_55,subset_120,subset_106,subset_110,subset_16,subset_27,subset_99,subset_26' '85344'
distillation 'subset_129,subset_35,subset_37,subset_148,subset_164,subset_12,subset_6,subset_197,subset_102,subset_180,subset_77,subset_161,subset_113,subset_188' 'subset_97,subset_6,subset_111,subset_63,subset_0,subset_83,subset_115,subset_74,subset_1' '27631'
distillation 'subset_125,subset_90,subset_106,subset_165,subset_209,subset_78,subset_199,subset_66,subset_87,subset_185,subset_136,subset_203,subset_24,subset_127' 'subset_75,subset_70,subset_47,subset_9,subset_23,subset_35,subset_42,subset_44,subset_105' '74991'
distillation 'subset_192,subset_193,subset_219,subset_19,subset_101,subset_218,subset_121,subset_151,subset_198,subset_176,subset_139,subset_147,subset_25,subset_95' 'subset_17,subset_66,subset_84,subset_76,subset_92,subset_96,subset_73,subset_5,subset_121' '54602'
distillation 'subset_130,subset_79,subset_220,subset_187,subset_97,subset_40,subset_115,subset_70,subset_215,subset_183,subset_119,subset_214,subset_111,subset_179' 'subset_128,subset_50,subset_123,subset_37,subset_40,subset_59,subset_137,subset_51,subset_7' '81016'
distillation 'subset_171,subset_8,subset_64,subset_163,subset_98,subset_131,subset_191,subset_85,subset_190,subset_55,subset_82,subset_91,subset_157,subset_109' 'subset_14,subset_18,subset_28,subset_49,subset_68,subset_72,subset_118,subset_54,subset_48' '83753'
distillation 'subset_7,subset_152,subset_26,subset_213,subset_210,subset_54,subset_143,subset_75,subset_195,subset_126,subset_175,subset_118,subset_42,subset_9' 'subset_107,subset_80,subset_13,subset_127,subset_101,subset_102,subset_134,subset_53,subset_57' '3969'
distillation 'subset_169,subset_178,subset_71,subset_88,subset_156,subset_155,subset_34,subset_41,subset_138,subset_141,subset_80,subset_207,subset_31,subset_174' 'subset_8,subset_116,subset_88,subset_4,subset_43,subset_91,subset_22,subset_29,subset_33' '65278'
distillation 'subset_217,subset_142,subset_15,subset_43,subset_39,subset_45,subset_2,subset_150,subset_73,subset_36,subset_170,subset_153,subset_89,subset_49' 'subset_60,subset_69,subset_108,subset_78,subset_67,subset_31,subset_95,subset_135,subset_93' '82203'
# Epoch 7
distillation 'subset_41,subset_116,subset_14,subset_126,subset_91,subset_58,subset_181,subset_215,subset_185,subset_171,subset_106,subset_72,subset_117,subset_6' 'subset_41,subset_63,subset_49,subset_21,subset_12,subset_15,subset_110,subset_6,subset_91' '89925'
distillation 'subset_54,subset_13,subset_57,subset_130,subset_96,subset_92,subset_40,subset_127,subset_188,subset_131,subset_79,subset_213,subset_99,subset_120' 'subset_16,subset_25,subset_44,subset_109,subset_69,subset_52,subset_112,subset_3,subset_65' '65866'
distillation 'subset_104,subset_141,subset_71,subset_5,subset_139,subset_189,subset_118,subset_176,subset_35,subset_174,subset_202,subset_33,subset_53,subset_44' 'subset_20,subset_115,subset_13,subset_55,subset_130,subset_39,subset_132,subset_58,subset_54' '92216'
distillation 'subset_219,subset_93,subset_152,subset_47,subset_151,subset_198,subset_128,subset_77,subset_50,subset_182,subset_111,subset_1,subset_123,subset_26' 'subset_106,subset_62,subset_87,subset_35,subset_36,subset_7,subset_135,subset_29,subset_48' '40760'
distillation 'subset_28,subset_216,subset_42,subset_90,subset_136,subset_18,subset_2,subset_107,subset_7,subset_46,subset_39,subset_22,subset_184,subset_80' 'subset_123,subset_116,subset_66,subset_26,subset_89,subset_37,subset_105,subset_30,subset_78' '90477'
distillation 'subset_31,subset_211,subset_204,subset_186,subset_81,subset_64,subset_150,subset_97,subset_98,subset_24,subset_179,subset_100,subset_142,subset_144' 'subset_128,subset_90,subset_95,subset_81,subset_72,subset_137,subset_101,subset_60,subset_2' '15389'
distillation 'subset_143,subset_194,subset_23,subset_60,subset_121,subset_222,subset_140,subset_109,subset_73,subset_158,subset_25,subset_161,subset_220,subset_208' 'subset_126,subset_82,subset_120,subset_80,subset_38,subset_57,subset_96,subset_108,subset_114' '83717'
distillation 'subset_32,subset_177,subset_137,subset_134,subset_200,subset_20,subset_149,subset_36,subset_191,subset_76,subset_160,subset_69,subset_166,subset_112' 'subset_14,subset_134,subset_45,subset_46,subset_75,subset_93,subset_86,subset_97,subset_61' '38671'
distillation 'subset_101,subset_221,subset_135,subset_85,subset_201,subset_148,subset_132,subset_103,subset_119,subset_30,subset_89,subset_190,subset_45,subset_172' 'subset_23,subset_127,subset_122,subset_56,subset_129,subset_43,subset_68,subset_31,subset_10' '48174'
distillation 'subset_197,subset_65,subset_169,subset_49,subset_84,subset_115,subset_146,subset_83,subset_94,subset_207,subset_114,subset_62,subset_10,subset_59' 'subset_24,subset_76,subset_103,subset_85,subset_22,subset_8,subset_33,subset_133,subset_84' '80560'
distillation 'subset_52,subset_105,subset_113,subset_34,subset_167,subset_147,subset_122,subset_87,subset_192,subset_205,subset_214,subset_175,subset_86,subset_183' 'subset_18,subset_67,subset_74,subset_104,subset_53,subset_73,subset_113,subset_27,subset_50' '28947'
distillation 'subset_56,subset_156,subset_78,subset_19,subset_37,subset_168,subset_124,subset_55,subset_38,subset_157,subset_88,subset_138,subset_193,subset_210' 'subset_98,subset_11,subset_131,subset_107,subset_77,subset_17,subset_94,subset_0,subset_92' '28721'
distillation 'subset_74,subset_15,subset_70,subset_159,subset_153,subset_9,subset_180,subset_163,subset_129,subset_27,subset_11,subset_48,subset_43,subset_21' 'subset_100,subset_32,subset_71,subset_9,subset_40,subset_42,subset_119,subset_125,subset_4' '17507'
distillation 'subset_165,subset_212,subset_4,subset_66,subset_51,subset_217,subset_206,subset_187,subset_218,subset_17,subset_61,subset_164,subset_3,subset_155' 'subset_121,subset_1,subset_88,subset_136,subset_83,subset_118,subset_19,subset_64,subset_99' '62676'
distillation 'subset_173,subset_178,subset_63,subset_68,subset_8,subset_108,subset_145,subset_199,subset_154,subset_102,subset_125,subset_29,subset_67,subset_195' 'subset_59,subset_111,subset_102,subset_34,subset_117,subset_70,subset_124,subset_47,subset_79' '20078'
# Epoch 8
distillation 'subset_112,subset_59,subset_6,subset_27,subset_114,subset_56,subset_18,subset_211,subset_33,subset_148,subset_31,subset_146,subset_90,subset_213' 'subset_111,subset_123,subset_13,subset_25,subset_100,subset_94,subset_82,subset_112,subset_37' '92123'
distillation 'subset_204,subset_127,subset_11,subset_159,subset_15,subset_145,subset_116,subset_97,subset_142,subset_35,subset_198,subset_93,subset_200,subset_100' 'subset_22,subset_137,subset_10,subset_66,subset_72,subset_4,subset_52,subset_132,subset_43' '22463'
distillation 'subset_106,subset_109,subset_218,subset_214,subset_30,subset_177,subset_124,subset_89,subset_179,subset_82,subset_22,subset_2,subset_24,subset_60' 'subset_77,subset_102,subset_130,subset_113,subset_91,subset_0,subset_73,subset_9,subset_61' '10837'
distillation 'subset_77,subset_69,subset_162,subset_81,subset_187,subset_19,subset_210,subset_85,subset_171,subset_194,subset_66,subset_209,subset_183,subset_9' 'subset_98,subset_3,subset_126,subset_120,subset_33,subset_44,subset_106,subset_80,subset_19' '65517'
distillation 'subset_50,subset_4,subset_217,subset_64,subset_136,subset_38,subset_126,subset_67,subset_190,subset_188,subset_51,subset_105,subset_25,subset_21' 'subset_17,subset_78,subset_134,subset_1,subset_64,subset_124,subset_28,subset_48,subset_29' '56873'
distillation 'subset_195,subset_39,subset_37,subset_206,subset_157,subset_44,subset_87,subset_53,subset_168,subset_158,subset_174,subset_7,subset_175,subset_215' 'subset_136,subset_63,subset_84,subset_101,subset_51,subset_125,subset_104,subset_71,subset_59' '84529'
distillation 'subset_120,subset_45,subset_103,subset_28,subset_133,subset_61,subset_65,subset_181,subset_140,subset_197,subset_166,subset_153,subset_78,subset_169' 'subset_21,subset_127,subset_133,subset_85,subset_69,subset_46,subset_93,subset_68,subset_18' '43420'
distillation 'subset_117,subset_68,subset_99,subset_8,subset_14,subset_32,subset_138,subset_1,subset_143,subset_20,subset_91,subset_74,subset_113,subset_110' 'subset_50,subset_129,subset_32,subset_41,subset_110,subset_16,subset_12,subset_75,subset_6' '74167'
distillation 'subset_101,subset_79,subset_176,subset_178,subset_152,subset_42,subset_5,subset_212,subset_36,subset_128,subset_184,subset_123,subset_80,subset_63' 'subset_56,subset_39,subset_24,subset_86,subset_65,subset_31,subset_128,subset_55,subset_38' '12499'
distillation 'subset_23,subset_193,subset_150,subset_88,subset_55,subset_216,subset_139,subset_137,subset_47,subset_71,subset_222,subset_147,subset_199,subset_125' 'subset_8,subset_70,subset_5,subset_7,subset_30,subset_103,subset_47,subset_83,subset_36' '69226'
distillation 'subset_220,subset_129,subset_208,subset_196,subset_76,subset_118,subset_151,subset_149,subset_186,subset_185,subset_0,subset_130,subset_3,subset_10' 'subset_119,subset_81,subset_87,subset_105,subset_58,subset_115,subset_116,subset_2,subset_121' '5577'
distillation 'subset_57,subset_121,subset_160,subset_72,subset_96,subset_192,subset_16,subset_41,subset_46,subset_134,subset_58,subset_98,subset_132,subset_34' 'subset_11,subset_122,subset_97,subset_99,subset_49,subset_20,subset_53,subset_62,subset_117' '30115'
distillation 'subset_163,subset_173,subset_165,subset_203,subset_115,subset_189,subset_95,subset_54,subset_154,subset_84,subset_131,subset_172,subset_205,subset_111' 'subset_15,subset_79,subset_42,subset_95,subset_54,subset_89,subset_92,subset_34,subset_96' '27710'
distillation 'subset_29,subset_141,subset_221,subset_83,subset_144,subset_219,subset_202,subset_12,subset_122,subset_182,subset_49,subset_170,subset_13,subset_104' 'subset_74,subset_107,subset_109,subset_57,subset_14,subset_40,subset_45,subset_90,subset_135' '90739'
distillation 'subset_167,subset_92,subset_73,subset_43,subset_207,subset_135,subset_94,subset_40,subset_108,subset_86,subset_75,subset_180,subset_107,subset_155' 'subset_26,subset_67,subset_88,subset_27,subset_23,subset_76,subset_108,subset_131,subset_60' '74289'

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


