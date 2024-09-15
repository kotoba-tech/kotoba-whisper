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

partion_size = 20
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
distillation 'subset_162,subset_36,subset_185,subset_2,subset_45,subset_105,subset_195,subset_168,subset_52,subset_208,subset_212' 'subset_67,subset_121,subset_126,subset_78,subset_134,subset_14' '17154'
distillation 'subset_159,subset_110,subset_123,subset_121,subset_98,subset_181,subset_60,subset_13,subset_44,subset_16,subset_84' 'subset_72,subset_113,subset_4,subset_26,subset_131,subset_38' '88039'
distillation 'subset_135,subset_29,subset_215,subset_21,subset_169,subset_194,subset_30,subset_127,subset_175,subset_4,subset_42' 'subset_74,subset_40,subset_75,subset_98,subset_117,subset_22' '84607'
distillation 'subset_47,subset_5,subset_77,subset_3,subset_15,subset_216,subset_190,subset_205,subset_81,subset_192,subset_99' 'subset_53,subset_47,subset_48,subset_64,subset_63,subset_25' '39321'
distillation 'subset_133,subset_66,subset_145,subset_73,subset_10,subset_122,subset_177,subset_9,subset_12,subset_38,subset_138' 'subset_19,subset_28,subset_5,subset_107,subset_2,subset_81' '59929'
distillation 'subset_61,subset_198,subset_89,subset_193,subset_134,subset_119,subset_79,subset_78,subset_209,subset_64,subset_103' 'subset_65,subset_36,subset_10,subset_33,subset_32,subset_127' '41441'
distillation 'subset_0,subset_111,subset_196,subset_180,subset_37,subset_222,subset_140,subset_126,subset_104,subset_144,subset_187' 'subset_41,subset_39,subset_20,subset_49,subset_34,subset_37' '98548'
distillation 'subset_218,subset_32,subset_171,subset_152,subset_85,subset_48,subset_161,subset_125,subset_76,subset_100,subset_153' 'subset_3,subset_91,subset_106,subset_76,subset_79,subset_80' '9508'
distillation 'subset_128,subset_19,subset_131,subset_199,subset_156,subset_115,subset_65,subset_184,subset_221,subset_46,subset_51' 'subset_105,subset_42,subset_97,subset_99,subset_46,subset_44' '1220'
distillation 'subset_101,subset_54,subset_172,subset_106,subset_109,subset_142,subset_149,subset_95,subset_167,subset_157,subset_33' 'subset_66,subset_129,subset_29,subset_133,subset_50,subset_122' '60068'
distillation 'subset_164,subset_130,subset_82,subset_113,subset_197,subset_63,subset_132,subset_124,subset_186,subset_174,subset_112' 'subset_87,subset_0,subset_88,subset_77,subset_95,subset_94' '81416'
distillation 'subset_72,subset_120,subset_148,subset_27,subset_201,subset_34,subset_102,subset_80,subset_206,subset_155,subset_14' 'subset_124,subset_61,subset_125,subset_120,subset_21,subset_30' '73792'
distillation 'subset_83,subset_191,subset_69,subset_211,subset_118,subset_146,subset_217,subset_136,subset_43,subset_18,subset_68' 'subset_11,subset_136,subset_123,subset_114,subset_96,subset_70' '13104'
distillation 'subset_53,subset_90,subset_94,subset_41,subset_93,subset_116,subset_182,subset_176,subset_25,subset_202,subset_165' 'subset_137,subset_9,subset_90,subset_102,subset_35,subset_23' '9602'
distillation 'subset_74,subset_58,subset_170,subset_17,subset_49,subset_147,subset_92,subset_158,subset_160,subset_75,subset_141' 'subset_104,subset_132,subset_57,subset_68,subset_89,subset_110' '70468'
distillation 'subset_20,subset_96,subset_31,subset_137,subset_117,subset_11,subset_67,subset_200,subset_88,subset_91,subset_24' 'subset_115,subset_13,subset_43,subset_101,subset_51,subset_7' '27938'
distillation 'subset_97,subset_204,subset_213,subset_86,subset_203,subset_39,subset_214,subset_87,subset_207,subset_178,subset_40' 'subset_108,subset_82,subset_83,subset_86,subset_6,subset_93' '66307'
distillation 'subset_1,subset_71,subset_150,subset_114,subset_56,subset_107,subset_210,subset_179,subset_166,subset_183,subset_50' 'subset_59,subset_111,subset_54,subset_45,subset_55,subset_130' '34760'
distillation 'subset_143,subset_220,subset_154,subset_129,subset_59,subset_55,subset_23,subset_7,subset_8,subset_108,subset_151' 'subset_109,subset_12,subset_24,subset_52,subset_103,subset_116' '17361'
distillation 'subset_22,subset_139,subset_219,subset_173,subset_26,subset_188,subset_35,subset_57,subset_62,subset_70,subset_189' 'subset_100,subset_31,subset_128,subset_118,subset_73,subset_112' '45745'
# Epoch 2
distillation 'subset_171,subset_69,subset_221,subset_163,subset_109,subset_39,subset_148,subset_89,subset_197,subset_147,subset_59' 'subset_37,subset_86,subset_53,subset_110,subset_116,subset_58' '2671'
distillation 'subset_144,subset_176,subset_138,subset_101,subset_36,subset_146,subset_202,subset_217,subset_102,subset_80,subset_168' 'subset_82,subset_78,subset_97,subset_38,subset_98,subset_84' '81439'
distillation 'subset_166,subset_4,subset_79,subset_177,subset_92,subset_181,subset_178,subset_96,subset_84,subset_25,subset_175' 'subset_24,subset_129,subset_117,subset_79,subset_40,subset_115' '19973'
distillation 'subset_152,subset_179,subset_161,subset_132,subset_27,subset_16,subset_200,subset_17,subset_14,subset_156,subset_68' 'subset_123,subset_1,subset_25,subset_2,subset_72,subset_49' '31266'
distillation 'subset_10,subset_106,subset_134,subset_28,subset_141,subset_3,subset_118,subset_204,subset_73,subset_99,subset_85' 'subset_8,subset_39,subset_31,subset_10,subset_114,subset_106' '16544'
distillation 'subset_213,subset_93,subset_110,subset_222,subset_167,subset_13,subset_19,subset_184,subset_49,subset_150,subset_72' 'subset_134,subset_15,subset_14,subset_51,subset_57,subset_91' '62070'
distillation 'subset_56,subset_88,subset_98,subset_128,subset_54,subset_137,subset_165,subset_145,subset_34,subset_174,subset_0' 'subset_99,subset_70,subset_93,subset_104,subset_28,subset_64' '87747'
distillation 'subset_97,subset_143,subset_105,subset_58,subset_108,subset_126,subset_122,subset_86,subset_207,subset_57,subset_41' 'subset_50,subset_121,subset_11,subset_60,subset_23,subset_69' '14993'
distillation 'subset_83,subset_95,subset_61,subset_7,subset_214,subset_45,subset_155,subset_104,subset_135,subset_91,subset_121' 'subset_85,subset_75,subset_20,subset_45,subset_101,subset_80' '73920'
distillation 'subset_63,subset_186,subset_191,subset_100,subset_62,subset_103,subset_48,subset_120,subset_173,subset_51,subset_52' 'subset_89,subset_136,subset_125,subset_103,subset_19,subset_94' '28569'
distillation 'subset_77,subset_192,subset_9,subset_215,subset_125,subset_2,subset_20,subset_154,subset_130,subset_172,subset_22' 'subset_9,subset_63,subset_52,subset_67,subset_0,subset_105' '60952'
distillation 'subset_94,subset_133,subset_193,subset_124,subset_90,subset_21,subset_198,subset_112,subset_158,subset_46,subset_220' 'subset_126,subset_111,subset_16,subset_5,subset_113,subset_92' '91680'
distillation 'subset_70,subset_196,subset_116,subset_188,subset_71,subset_43,subset_195,subset_149,subset_66,subset_67,subset_119' 'subset_62,subset_27,subset_68,subset_46,subset_44,subset_4' '33586'
distillation 'subset_183,subset_142,subset_139,subset_131,subset_55,subset_209,subset_107,subset_199,subset_210,subset_115,subset_32' 'subset_56,subset_35,subset_42,subset_81,subset_47,subset_22' '48351'
distillation 'subset_212,subset_42,subset_164,subset_15,subset_23,subset_182,subset_6,subset_208,subset_35,subset_216,subset_44' 'subset_131,subset_120,subset_119,subset_3,subset_71,subset_130' '21992'
distillation 'subset_30,subset_151,subset_53,subset_33,subset_50,subset_113,subset_81,subset_40,subset_75,subset_47,subset_76' 'subset_41,subset_13,subset_26,subset_118,subset_6,subset_65' '79415'
distillation 'subset_162,subset_159,subset_37,subset_157,subset_1,subset_29,subset_123,subset_64,subset_201,subset_206,subset_129' 'subset_90,subset_88,subset_124,subset_83,subset_127,subset_102' '79593'
distillation 'subset_160,subset_24,subset_12,subset_153,subset_87,subset_38,subset_74,subset_189,subset_180,subset_190,subset_219' 'subset_135,subset_112,subset_33,subset_12,subset_87,subset_7' '98032'
distillation 'subset_114,subset_194,subset_127,subset_111,subset_5,subset_169,subset_117,subset_187,subset_18,subset_11,subset_185' 'subset_48,subset_133,subset_21,subset_55,subset_66,subset_95' '94156'
distillation 'subset_211,subset_31,subset_8,subset_170,subset_218,subset_203,subset_136,subset_26,subset_82,subset_205,subset_140' 'subset_43,subset_54,subset_137,subset_18,subset_100,subset_108' '15012'
# Epoch 3
distillation 'subset_212,subset_108,subset_180,subset_201,subset_167,subset_165,subset_38,subset_40,subset_115,subset_145,subset_110' 'subset_23,subset_2,subset_118,subset_34,subset_8,subset_30' '50277'
distillation 'subset_12,subset_195,subset_171,subset_65,subset_58,subset_218,subset_3,subset_9,subset_67,subset_197,subset_99' 'subset_127,subset_82,subset_60,subset_120,subset_21,subset_17' '24983'
distillation 'subset_80,subset_47,subset_22,subset_63,subset_20,subset_143,subset_120,subset_217,subset_53,subset_45,subset_149' 'subset_109,subset_87,subset_86,subset_117,subset_121,subset_85' '78561'
distillation 'subset_51,subset_178,subset_207,subset_152,subset_101,subset_169,subset_170,subset_150,subset_186,subset_203,subset_185' 'subset_26,subset_43,subset_10,subset_134,subset_70,subset_38' '66616'
distillation 'subset_78,subset_44,subset_147,subset_131,subset_49,subset_8,subset_29,subset_187,subset_107,subset_177,subset_46' 'subset_56,subset_3,subset_89,subset_131,subset_1,subset_137' '97818'
distillation 'subset_73,subset_114,subset_111,subset_213,subset_190,subset_206,subset_72,subset_144,subset_68,subset_198,subset_62' 'subset_80,subset_107,subset_105,subset_96,subset_36,subset_97' '17892'
distillation 'subset_33,subset_81,subset_136,subset_138,subset_182,subset_127,subset_133,subset_215,subset_26,subset_39,subset_7' 'subset_47,subset_115,subset_92,subset_69,subset_114,subset_28' '9150'
distillation 'subset_146,subset_121,subset_69,subset_16,subset_157,subset_199,subset_48,subset_27,subset_153,subset_176,subset_132' 'subset_15,subset_122,subset_81,subset_51,subset_24,subset_124' '36208'
distillation 'subset_0,subset_5,subset_112,subset_161,subset_116,subset_123,subset_100,subset_24,subset_66,subset_42,subset_160' 'subset_31,subset_90,subset_41,subset_133,subset_37,subset_74' '54387'
distillation 'subset_220,subset_74,subset_17,subset_79,subset_1,subset_36,subset_96,subset_175,subset_87,subset_179,subset_222' 'subset_83,subset_35,subset_39,subset_75,subset_78,subset_63' '44548'
distillation 'subset_95,subset_60,subset_156,subset_23,subset_104,subset_82,subset_10,subset_88,subset_183,subset_13,subset_28' 'subset_40,subset_113,subset_54,subset_19,subset_116,subset_66' '66550'
distillation 'subset_34,subset_86,subset_159,subset_25,subset_189,subset_204,subset_93,subset_174,subset_4,subset_109,subset_125' 'subset_108,subset_29,subset_42,subset_95,subset_25,subset_129' '35018'
distillation 'subset_188,subset_94,subset_113,subset_31,subset_103,subset_19,subset_168,subset_141,subset_70,subset_221,subset_119' 'subset_59,subset_32,subset_101,subset_11,subset_13,subset_88' '336'
distillation 'subset_54,subset_208,subset_129,subset_56,subset_61,subset_85,subset_124,subset_134,subset_122,subset_21,subset_89' 'subset_52,subset_112,subset_68,subset_49,subset_91,subset_7' '37072'
distillation 'subset_57,subset_91,subset_164,subset_139,subset_130,subset_216,subset_2,subset_137,subset_166,subset_117,subset_126' 'subset_22,subset_126,subset_9,subset_72,subset_33,subset_135' '95164'
distillation 'subset_98,subset_11,subset_18,subset_172,subset_193,subset_43,subset_15,subset_128,subset_151,subset_196,subset_106' 'subset_55,subset_61,subset_53,subset_46,subset_27,subset_119' '39132'
distillation 'subset_192,subset_200,subset_154,subset_35,subset_214,subset_77,subset_205,subset_90,subset_173,subset_163,subset_41' 'subset_76,subset_84,subset_110,subset_14,subset_4,subset_64' '76931'
distillation 'subset_30,subset_158,subset_202,subset_155,subset_50,subset_52,subset_71,subset_83,subset_59,subset_142,subset_84' 'subset_6,subset_136,subset_58,subset_132,subset_98,subset_100' '76011'
distillation 'subset_32,subset_76,subset_97,subset_219,subset_37,subset_92,subset_184,subset_6,subset_162,subset_210,subset_102' 'subset_67,subset_5,subset_0,subset_57,subset_45,subset_130' '86499'
distillation 'subset_118,subset_64,subset_191,subset_135,subset_75,subset_55,subset_140,subset_148,subset_209,subset_181,subset_105' 'subset_73,subset_71,subset_94,subset_128,subset_125,subset_77' '64178'
# Epoch 4
distillation 'subset_143,subset_69,subset_16,subset_80,subset_107,subset_124,subset_139,subset_102,subset_125,subset_64,subset_205' 'subset_68,subset_47,subset_132,subset_30,subset_66,subset_6' '52262'
distillation 'subset_106,subset_186,subset_132,subset_73,subset_26,subset_136,subset_187,subset_134,subset_58,subset_177,subset_179' 'subset_71,subset_92,subset_87,subset_134,subset_110,subset_20' '54524'
distillation 'subset_84,subset_88,subset_137,subset_183,subset_55,subset_111,subset_140,subset_45,subset_33,subset_43,subset_222' 'subset_50,subset_37,subset_136,subset_120,subset_129,subset_73' '95631'
distillation 'subset_85,subset_220,subset_212,subset_150,subset_182,subset_83,subset_92,subset_74,subset_208,subset_89,subset_151' 'subset_93,subset_85,subset_52,subset_10,subset_112,subset_54' '12463'
distillation 'subset_121,subset_61,subset_99,subset_9,subset_135,subset_62,subset_110,subset_129,subset_116,subset_51,subset_57' 'subset_111,subset_114,subset_122,subset_29,subset_21,subset_81' '40966'
distillation 'subset_178,subset_14,subset_115,subset_127,subset_198,subset_20,subset_131,subset_173,subset_112,subset_162,subset_218' 'subset_75,subset_125,subset_31,subset_46,subset_74,subset_126' '55936'
distillation 'subset_105,subset_167,subset_70,subset_192,subset_165,subset_195,subset_15,subset_1,subset_22,subset_120,subset_123' 'subset_123,subset_35,subset_61,subset_12,subset_64,subset_43' '40963'
distillation 'subset_175,subset_217,subset_77,subset_68,subset_146,subset_38,subset_133,subset_196,subset_207,subset_29,subset_200' 'subset_63,subset_7,subset_108,subset_16,subset_32,subset_133' '87177'
distillation 'subset_180,subset_119,subset_5,subset_75,subset_100,subset_44,subset_164,subset_118,subset_53,subset_171,subset_109' 'subset_83,subset_19,subset_115,subset_9,subset_38,subset_90' '33408'
distillation 'subset_8,subset_30,subset_87,subset_181,subset_86,subset_209,subset_46,subset_189,subset_52,subset_54,subset_94' 'subset_1,subset_42,subset_65,subset_113,subset_67,subset_116' '49060'
distillation 'subset_114,subset_117,subset_188,subset_138,subset_128,subset_158,subset_13,subset_50,subset_79,subset_82,subset_108' 'subset_130,subset_45,subset_135,subset_119,subset_96,subset_101' '20006'
distillation 'subset_148,subset_166,subset_91,subset_202,subset_126,subset_4,subset_184,subset_35,subset_93,subset_210,subset_98' 'subset_76,subset_28,subset_128,subset_106,subset_2,subset_88' '90023'
distillation 'subset_172,subset_78,subset_185,subset_147,subset_37,subset_31,subset_144,subset_101,subset_95,subset_24,subset_161' 'subset_34,subset_86,subset_70,subset_105,subset_57,subset_53' '62163'
distillation 'subset_39,subset_65,subset_48,subset_219,subset_40,subset_7,subset_203,subset_23,subset_21,subset_90,subset_60' 'subset_48,subset_99,subset_23,subset_84,subset_26,subset_103' '8799'
distillation 'subset_47,subset_213,subset_215,subset_206,subset_3,subset_130,subset_76,subset_168,subset_25,subset_2,subset_97' 'subset_137,subset_4,subset_11,subset_55,subset_25,subset_60' '11957'
distillation 'subset_142,subset_103,subset_41,subset_197,subset_71,subset_17,subset_216,subset_176,subset_81,subset_28,subset_67' 'subset_89,subset_59,subset_121,subset_107,subset_15,subset_127' '11188'
distillation 'subset_170,subset_18,subset_36,subset_152,subset_201,subset_191,subset_113,subset_163,subset_156,subset_63,subset_27' 'subset_118,subset_13,subset_49,subset_0,subset_102,subset_22' '12219'
distillation 'subset_122,subset_174,subset_149,subset_145,subset_204,subset_194,subset_11,subset_193,subset_6,subset_42,subset_214' 'subset_17,subset_78,subset_117,subset_56,subset_44,subset_41' '56606'
distillation 'subset_34,subset_155,subset_157,subset_12,subset_96,subset_32,subset_190,subset_160,subset_56,subset_72,subset_154' 'subset_82,subset_51,subset_91,subset_95,subset_79,subset_72' '12656'
distillation 'subset_49,subset_199,subset_10,subset_66,subset_141,subset_59,subset_221,subset_153,subset_0,subset_159,subset_19' 'subset_109,subset_24,subset_5,subset_80,subset_77,subset_39' '97594'
# Epoch 5
distillation 'subset_108,subset_105,subset_103,subset_80,subset_175,subset_125,subset_76,subset_131,subset_86,subset_220,subset_185' 'subset_79,subset_68,subset_30,subset_51,subset_91,subset_22' '84040'
distillation 'subset_163,subset_3,subset_96,subset_144,subset_203,subset_20,subset_160,subset_117,subset_116,subset_165,subset_151' 'subset_47,subset_116,subset_18,subset_117,subset_9,subset_42' '62991'
distillation 'subset_24,subset_37,subset_68,subset_33,subset_79,subset_153,subset_134,subset_137,subset_198,subset_218,subset_212' 'subset_106,subset_14,subset_126,subset_103,subset_131,subset_120' '48655'
distillation 'subset_25,subset_188,subset_67,subset_87,subset_214,subset_89,subset_170,subset_209,subset_7,subset_184,subset_121' 'subset_114,subset_75,subset_73,subset_0,subset_29,subset_81' '72684'
distillation 'subset_8,subset_18,subset_210,subset_11,subset_158,subset_38,subset_47,subset_107,subset_205,subset_62,subset_14' 'subset_105,subset_64,subset_45,subset_101,subset_124,subset_56' '13446'
distillation 'subset_31,subset_100,subset_9,subset_58,subset_44,subset_183,subset_90,subset_19,subset_22,subset_166,subset_10' 'subset_127,subset_135,subset_122,subset_16,subset_21,subset_35' '93244'
distillation 'subset_34,subset_126,subset_106,subset_142,subset_141,subset_63,subset_199,subset_75,subset_217,subset_73,subset_201' 'subset_97,subset_4,subset_118,subset_111,subset_61,subset_90' '67621'
distillation 'subset_51,subset_129,subset_146,subset_42,subset_32,subset_113,subset_41,subset_127,subset_16,subset_30,subset_82' 'subset_2,subset_67,subset_128,subset_119,subset_46,subset_74' '16341'
distillation 'subset_50,subset_64,subset_61,subset_148,subset_27,subset_162,subset_172,subset_157,subset_81,subset_135,subset_136' 'subset_5,subset_121,subset_107,subset_66,subset_134,subset_25' '37363'
distillation 'subset_84,subset_78,subset_169,subset_112,subset_128,subset_0,subset_208,subset_182,subset_222,subset_171,subset_168' 'subset_40,subset_36,subset_82,subset_137,subset_10,subset_39' '10993'
distillation 'subset_109,subset_204,subset_83,subset_155,subset_143,subset_191,subset_123,subset_110,subset_35,subset_124,subset_2' 'subset_98,subset_109,subset_27,subset_132,subset_102,subset_1' '21012'
distillation 'subset_207,subset_167,subset_173,subset_120,subset_4,subset_28,subset_15,subset_140,subset_114,subset_180,subset_122' 'subset_20,subset_34,subset_99,subset_58,subset_84,subset_17' '35755'
distillation 'subset_98,subset_215,subset_95,subset_189,subset_101,subset_202,subset_56,subset_138,subset_154,subset_177,subset_186' 'subset_129,subset_115,subset_12,subset_108,subset_93,subset_70' '58893'
distillation 'subset_21,subset_6,subset_190,subset_179,subset_48,subset_133,subset_71,subset_174,subset_74,subset_200,subset_13' 'subset_72,subset_55,subset_33,subset_62,subset_31,subset_69' '67286'
distillation 'subset_52,subset_53,subset_178,subset_147,subset_88,subset_152,subset_36,subset_187,subset_181,subset_65,subset_150' 'subset_13,subset_88,subset_50,subset_60,subset_87,subset_63' '19313'
distillation 'subset_66,subset_69,subset_145,subset_29,subset_12,subset_192,subset_99,subset_102,subset_70,subset_193,subset_213' 'subset_95,subset_23,subset_71,subset_57,subset_11,subset_80' '57340'
distillation 'subset_139,subset_211,subset_97,subset_176,subset_206,subset_130,subset_59,subset_94,subset_104,subset_219,subset_195' 'subset_38,subset_19,subset_123,subset_133,subset_32,subset_52' '12021'
distillation 'subset_45,subset_164,subset_159,subset_119,subset_111,subset_115,subset_91,subset_197,subset_92,subset_57,subset_93' 'subset_48,subset_76,subset_28,subset_44,subset_65,subset_89' '29102'
distillation 'subset_40,subset_55,subset_49,subset_77,subset_60,subset_1,subset_132,subset_156,subset_54,subset_194,subset_17' 'subset_125,subset_6,subset_112,subset_26,subset_8,subset_100' '59130'
distillation 'subset_5,subset_46,subset_43,subset_216,subset_196,subset_221,subset_39,subset_23,subset_26,subset_161,subset_85' 'subset_59,subset_7,subset_41,subset_104,subset_78,subset_37' '45820'
# Epoch 6
distillation 'subset_99,subset_25,subset_36,subset_137,subset_130,subset_123,subset_88,subset_101,subset_205,subset_176,subset_35' 'subset_87,subset_13,subset_133,subset_105,subset_67,subset_97' '90257'
distillation 'subset_19,subset_55,subset_30,subset_63,subset_198,subset_74,subset_146,subset_126,subset_144,subset_172,subset_213' 'subset_6,subset_26,subset_50,subset_69,subset_2,subset_91' '7852'
distillation 'subset_40,subset_102,subset_211,subset_180,subset_12,subset_191,subset_178,subset_43,subset_135,subset_109,subset_158' 'subset_5,subset_90,subset_121,subset_19,subset_38,subset_14' '20719'
distillation 'subset_129,subset_156,subset_115,subset_62,subset_162,subset_199,subset_112,subset_71,subset_80,subset_94,subset_204' 'subset_110,subset_47,subset_113,subset_123,subset_77,subset_101' '57619'
distillation 'subset_145,subset_39,subset_81,subset_210,subset_103,subset_93,subset_207,subset_216,subset_8,subset_106,subset_153' 'subset_30,subset_79,subset_11,subset_68,subset_65,subset_56' '54476'
distillation 'subset_98,subset_45,subset_192,subset_116,subset_117,subset_6,subset_111,subset_78,subset_72,subset_7,subset_208' 'subset_45,subset_0,subset_112,subset_98,subset_93,subset_27' '63471'
distillation 'subset_201,subset_46,subset_141,subset_14,subset_185,subset_127,subset_34,subset_70,subset_73,subset_22,subset_155' 'subset_33,subset_24,subset_54,subset_31,subset_80,subset_35' '60875'
distillation 'subset_120,subset_38,subset_100,subset_107,subset_164,subset_29,subset_203,subset_161,subset_184,subset_41,subset_149' 'subset_129,subset_106,subset_102,subset_22,subset_83,subset_94' '26741'
distillation 'subset_28,subset_66,subset_68,subset_167,subset_195,subset_212,subset_57,subset_4,subset_47,subset_159,subset_170' 'subset_48,subset_126,subset_119,subset_89,subset_55,subset_9' '44589'
distillation 'subset_209,subset_83,subset_173,subset_85,subset_186,subset_5,subset_132,subset_104,subset_58,subset_124,subset_193' 'subset_75,subset_17,subset_16,subset_116,subset_25,subset_71' '79516'
distillation 'subset_86,subset_215,subset_128,subset_37,subset_42,subset_150,subset_64,subset_44,subset_82,subset_56,subset_13' 'subset_57,subset_10,subset_122,subset_36,subset_99,subset_81' '18830'
distillation 'subset_24,subset_113,subset_0,subset_121,subset_142,subset_18,subset_52,subset_188,subset_54,subset_79,subset_219' 'subset_108,subset_130,subset_86,subset_37,subset_43,subset_12' '40970'
distillation 'subset_50,subset_133,subset_148,subset_194,subset_166,subset_196,subset_21,subset_163,subset_92,subset_84,subset_125' 'subset_104,subset_41,subset_117,subset_28,subset_70,subset_134' '94157'
distillation 'subset_177,subset_190,subset_114,subset_138,subset_174,subset_160,subset_27,subset_95,subset_151,subset_11,subset_118' 'subset_15,subset_109,subset_107,subset_111,subset_100,subset_92' '41853'
distillation 'subset_179,subset_181,subset_61,subset_91,subset_108,subset_168,subset_217,subset_87,subset_139,subset_48,subset_122' 'subset_72,subset_131,subset_44,subset_39,subset_74,subset_3' '96233'
distillation 'subset_187,subset_119,subset_202,subset_200,subset_15,subset_17,subset_20,subset_110,subset_152,subset_65,subset_222' 'subset_8,subset_32,subset_51,subset_132,subset_136,subset_120' '45268'
distillation 'subset_90,subset_3,subset_218,subset_51,subset_23,subset_77,subset_134,subset_171,subset_189,subset_53,subset_154' 'subset_64,subset_103,subset_124,subset_1,subset_46,subset_52' '52260'
distillation 'subset_89,subset_31,subset_10,subset_175,subset_221,subset_105,subset_147,subset_143,subset_49,subset_206,subset_197' 'subset_49,subset_62,subset_63,subset_85,subset_96,subset_73' '17139'
distillation 'subset_1,subset_97,subset_2,subset_75,subset_220,subset_67,subset_214,subset_60,subset_16,subset_69,subset_59' 'subset_78,subset_118,subset_4,subset_58,subset_84,subset_18' '99701'
distillation 'subset_33,subset_32,subset_131,subset_9,subset_26,subset_165,subset_136,subset_183,subset_157,subset_140,subset_169' 'subset_95,subset_128,subset_88,subset_61,subset_125,subset_60' '48567'
# Epoch 7
distillation 'subset_101,subset_151,subset_60,subset_108,subset_192,subset_220,subset_10,subset_25,subset_180,subset_99,subset_3' 'subset_28,subset_75,subset_111,subset_109,subset_47,subset_83' '59634'
distillation 'subset_171,subset_65,subset_175,subset_66,subset_14,subset_162,subset_17,subset_47,subset_116,subset_133,subset_185' 'subset_12,subset_102,subset_95,subset_62,subset_72,subset_8' '98007'
distillation 'subset_215,subset_117,subset_128,subset_194,subset_95,subset_178,subset_82,subset_1,subset_218,subset_120,subset_216' 'subset_0,subset_85,subset_133,subset_88,subset_117,subset_105' '79395'
distillation 'subset_161,subset_169,subset_146,subset_198,subset_23,subset_21,subset_159,subset_74,subset_40,subset_145,subset_29' 'subset_1,subset_49,subset_126,subset_131,subset_103,subset_59' '48971'
distillation 'subset_200,subset_173,subset_148,subset_42,subset_124,subset_15,subset_160,subset_222,subset_113,subset_221,subset_87' 'subset_11,subset_44,subset_16,subset_69,subset_2,subset_135' '54505'
distillation 'subset_193,subset_205,subset_39,subset_61,subset_81,subset_187,subset_153,subset_75,subset_167,subset_201,subset_12' 'subset_98,subset_65,subset_17,subset_73,subset_38,subset_58' '92004'
distillation 'subset_8,subset_144,subset_209,subset_0,subset_45,subset_132,subset_138,subset_217,subset_149,subset_130,subset_183' 'subset_19,subset_53,subset_45,subset_50,subset_89,subset_6' '71989'
distillation 'subset_179,subset_97,subset_100,subset_142,subset_90,subset_213,subset_53,subset_190,subset_88,subset_43,subset_181' 'subset_108,subset_23,subset_97,subset_81,subset_52,subset_40' '61712'
distillation 'subset_118,subset_67,subset_129,subset_37,subset_155,subset_137,subset_2,subset_28,subset_69,subset_112,subset_33' 'subset_78,subset_80,subset_31,subset_4,subset_42,subset_48' '99267'
distillation 'subset_106,subset_63,subset_168,subset_94,subset_210,subset_20,subset_5,subset_6,subset_154,subset_51,subset_134' 'subset_106,subset_107,subset_110,subset_74,subset_123,subset_94' '70453'
distillation 'subset_78,subset_136,subset_208,subset_104,subset_7,subset_204,subset_22,subset_166,subset_16,subset_57,subset_121' 'subset_61,subset_46,subset_22,subset_112,subset_70,subset_128' '87044'
distillation 'subset_163,subset_172,subset_197,subset_105,subset_143,subset_80,subset_50,subset_199,subset_64,subset_24,subset_184' 'subset_15,subset_121,subset_30,subset_93,subset_21,subset_99' '28627'
distillation 'subset_115,subset_176,subset_48,subset_35,subset_188,subset_139,subset_79,subset_207,subset_9,subset_122,subset_125' 'subset_125,subset_76,subset_130,subset_20,subset_25,subset_63' '99858'
distillation 'subset_41,subset_4,subset_31,subset_131,subset_158,subset_135,subset_127,subset_92,subset_165,subset_89,subset_59' 'subset_7,subset_114,subset_134,subset_104,subset_54,subset_13' '32435'
distillation 'subset_126,subset_152,subset_84,subset_195,subset_93,subset_182,subset_114,subset_30,subset_85,subset_19,subset_119' 'subset_3,subset_24,subset_101,subset_36,subset_119,subset_132' '89173'
distillation 'subset_186,subset_212,subset_46,subset_156,subset_86,subset_96,subset_36,subset_196,subset_70,subset_13,subset_110' 'subset_120,subset_9,subset_77,subset_67,subset_51,subset_136' '98925'
distillation 'subset_202,subset_11,subset_18,subset_34,subset_77,subset_32,subset_68,subset_27,subset_98,subset_140,subset_44' 'subset_86,subset_56,subset_41,subset_118,subset_91,subset_60' '78094'
distillation 'subset_54,subset_83,subset_102,subset_123,subset_147,subset_38,subset_206,subset_157,subset_76,subset_52,subset_214' 'subset_26,subset_100,subset_96,subset_35,subset_33,subset_64' '10730'
distillation 'subset_49,subset_103,subset_107,subset_189,subset_174,subset_203,subset_71,subset_26,subset_58,subset_91,subset_170' 'subset_79,subset_18,subset_115,subset_82,subset_84,subset_92' '68879'
distillation 'subset_73,subset_55,subset_211,subset_62,subset_72,subset_150,subset_141,subset_109,subset_56,subset_111,subset_164' 'subset_29,subset_113,subset_43,subset_87,subset_14,subset_124' '58556'
# Epoch 8
distillation 'subset_83,subset_151,subset_194,subset_98,subset_90,subset_202,subset_48,subset_32,subset_204,subset_138,subset_160' 'subset_48,subset_1,subset_25,subset_82,subset_137,subset_118' '2052'
distillation 'subset_42,subset_37,subset_84,subset_198,subset_170,subset_195,subset_18,subset_213,subset_75,subset_54,subset_210' 'subset_6,subset_14,subset_11,subset_132,subset_76,subset_122' '27152'
distillation 'subset_94,subset_58,subset_163,subset_211,subset_81,subset_3,subset_122,subset_72,subset_125,subset_186,subset_209' 'subset_107,subset_45,subset_2,subset_16,subset_52,subset_109' '76943'
distillation 'subset_172,subset_60,subset_15,subset_212,subset_150,subset_105,subset_152,subset_29,subset_126,subset_179,subset_26' 'subset_72,subset_85,subset_74,subset_130,subset_39,subset_91' '18899'
distillation 'subset_114,subset_185,subset_175,subset_70,subset_36,subset_128,subset_103,subset_206,subset_34,subset_50,subset_80' 'subset_26,subset_126,subset_99,subset_66,subset_94,subset_98' '98599'
distillation 'subset_88,subset_108,subset_68,subset_208,subset_99,subset_89,subset_100,subset_134,subset_107,subset_92,subset_187' 'subset_80,subset_89,subset_38,subset_128,subset_32,subset_114' '93416'
distillation 'subset_49,subset_149,subset_85,subset_164,subset_181,subset_221,subset_33,subset_144,subset_115,subset_200,subset_123' 'subset_133,subset_31,subset_87,subset_127,subset_20,subset_124' '52205'
distillation 'subset_165,subset_91,subset_87,subset_142,subset_215,subset_130,subset_132,subset_5,subset_146,subset_191,subset_174' 'subset_83,subset_129,subset_113,subset_103,subset_117,subset_61' '10084'
distillation 'subset_38,subset_8,subset_190,subset_11,subset_73,subset_59,subset_40,subset_57,subset_86,subset_147,subset_74' 'subset_50,subset_0,subset_88,subset_119,subset_73,subset_106' '39226'
distillation 'subset_46,subset_137,subset_78,subset_23,subset_201,subset_96,subset_25,subset_106,subset_93,subset_178,subset_0' 'subset_101,subset_10,subset_4,subset_63,subset_92,subset_97' '21292'
distillation 'subset_71,subset_205,subset_141,subset_31,subset_2,subset_120,subset_10,subset_136,subset_129,subset_43,subset_197' 'subset_120,subset_84,subset_71,subset_19,subset_58,subset_43' '73919'
distillation 'subset_158,subset_199,subset_102,subset_65,subset_214,subset_95,subset_153,subset_192,subset_19,subset_156,subset_56' 'subset_7,subset_3,subset_30,subset_29,subset_111,subset_136' '31519'
distillation 'subset_124,subset_45,subset_159,subset_203,subset_166,subset_167,subset_61,subset_30,subset_188,subset_110,subset_207' 'subset_46,subset_9,subset_40,subset_96,subset_70,subset_69' '74277'
distillation 'subset_44,subset_183,subset_189,subset_177,subset_39,subset_216,subset_219,subset_63,subset_55,subset_173,subset_218' 'subset_75,subset_55,subset_95,subset_57,subset_27,subset_41' '51007'
distillation 'subset_67,subset_119,subset_140,subset_133,subset_182,subset_69,subset_52,subset_112,subset_28,subset_154,subset_169' 'subset_37,subset_28,subset_12,subset_24,subset_62,subset_116' '88740'
distillation 'subset_148,subset_20,subset_217,subset_180,subset_121,subset_17,subset_109,subset_79,subset_162,subset_118,subset_51' 'subset_121,subset_102,subset_64,subset_86,subset_59,subset_115' '70852'
distillation 'subset_171,subset_47,subset_135,subset_193,subset_101,subset_12,subset_168,subset_220,subset_97,subset_196,subset_16' 'subset_93,subset_79,subset_105,subset_100,subset_81,subset_60' '43480'
distillation 'subset_41,subset_76,subset_111,subset_157,subset_6,subset_184,subset_161,subset_22,subset_66,subset_7,subset_24' 'subset_67,subset_131,subset_123,subset_18,subset_34,subset_54' '50311'
distillation 'subset_117,subset_143,subset_127,subset_27,subset_113,subset_13,subset_1,subset_104,subset_176,subset_53,subset_145' 'subset_77,subset_22,subset_68,subset_56,subset_17,subset_23' '99039'
distillation 'subset_21,subset_139,subset_131,subset_222,subset_64,subset_9,subset_62,subset_14,subset_82,subset_4,subset_116' 'subset_36,subset_13,subset_21,subset_112,subset_65,subset_90' '96959'


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


