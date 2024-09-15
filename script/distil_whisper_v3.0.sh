# kotoba-whisper-v2 distillation version
##########
# Config #
##########
WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts.
HF_MODEL_ALIAS="distil-whisper-bilingual"  # Model alias used when pushing models.
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
    --model_name_or_path "./${HF_MODEL_ALIAS}-init" \
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
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_workers 64 \
    --dataloader_num_workers 1 \
    --output_dir "./" \
    --wandb_project "wandb.${HF_MODEL_ALIAS}" \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --seed ${SEED} \
    --report_to "none" \
    --num_train_epochs 1
}
```python
from random import shuffle, seed, randint

partion_size = 10
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
distillation 'subset_162,subset_36,subset_185,subset_2,subset_45,subset_105,subset_195,subset_168,subset_52,subset_208,subset_212,subset_159,subset_110,subset_123,subset_121,subset_98,subset_181,subset_60,subset_13,subset_44,subset_16,subset_84' 'subset_67,subset_121,subset_126,subset_78,subset_134,subset_14,subset_72,subset_113,subset_4,subset_26,subset_131,subset_38,subset_74' '17154'
distillation 'subset_135,subset_29,subset_215,subset_21,subset_169,subset_194,subset_30,subset_127,subset_175,subset_4,subset_42,subset_47,subset_5,subset_77,subset_3,subset_15,subset_216,subset_190,subset_205,subset_81,subset_192,subset_99' 'subset_40,subset_75,subset_98,subset_117,subset_22,subset_53,subset_47,subset_48,subset_64,subset_63,subset_25,subset_19,subset_28' '88039'
distillation 'subset_133,subset_66,subset_145,subset_73,subset_10,subset_122,subset_177,subset_9,subset_12,subset_38,subset_138,subset_61,subset_198,subset_89,subset_193,subset_134,subset_119,subset_79,subset_78,subset_209,subset_64,subset_103' 'subset_5,subset_107,subset_2,subset_81,subset_65,subset_36,subset_10,subset_33,subset_32,subset_127,subset_41,subset_39,subset_20' '84607'
distillation 'subset_0,subset_111,subset_196,subset_180,subset_37,subset_222,subset_140,subset_126,subset_104,subset_144,subset_187,subset_218,subset_32,subset_171,subset_152,subset_85,subset_48,subset_161,subset_125,subset_76,subset_100,subset_153' 'subset_49,subset_34,subset_37,subset_3,subset_91,subset_106,subset_76,subset_79,subset_80,subset_105,subset_42,subset_97,subset_99' '39321'
distillation 'subset_128,subset_19,subset_131,subset_199,subset_156,subset_115,subset_65,subset_184,subset_221,subset_46,subset_51,subset_101,subset_54,subset_172,subset_106,subset_109,subset_142,subset_149,subset_95,subset_167,subset_157,subset_33' 'subset_46,subset_44,subset_66,subset_129,subset_29,subset_133,subset_50,subset_122,subset_87,subset_0,subset_88,subset_77,subset_95' '59929'
distillation 'subset_164,subset_130,subset_82,subset_113,subset_197,subset_63,subset_132,subset_124,subset_186,subset_174,subset_112,subset_72,subset_120,subset_148,subset_27,subset_201,subset_34,subset_102,subset_80,subset_206,subset_155,subset_14' 'subset_94,subset_124,subset_61,subset_125,subset_120,subset_21,subset_30,subset_11,subset_136,subset_123,subset_114,subset_96,subset_70' '41441'
distillation 'subset_83,subset_191,subset_69,subset_211,subset_118,subset_146,subset_217,subset_136,subset_43,subset_18,subset_68,subset_53,subset_90,subset_94,subset_41,subset_93,subset_116,subset_182,subset_176,subset_25,subset_202,subset_165' 'subset_137,subset_9,subset_90,subset_102,subset_35,subset_23,subset_104,subset_132,subset_57,subset_68,subset_89,subset_110,subset_115' '98548'
distillation 'subset_74,subset_58,subset_170,subset_17,subset_49,subset_147,subset_92,subset_158,subset_160,subset_75,subset_141,subset_20,subset_96,subset_31,subset_137,subset_117,subset_11,subset_67,subset_200,subset_88,subset_91,subset_24' 'subset_13,subset_43,subset_101,subset_51,subset_7,subset_108,subset_82,subset_83,subset_86,subset_6,subset_93,subset_59,subset_111' '9508'
distillation 'subset_97,subset_204,subset_213,subset_86,subset_203,subset_39,subset_214,subset_87,subset_207,subset_178,subset_40,subset_1,subset_71,subset_150,subset_114,subset_56,subset_107,subset_210,subset_179,subset_166,subset_183,subset_50' 'subset_54,subset_45,subset_55,subset_130,subset_109,subset_12,subset_24,subset_52,subset_103,subset_116,subset_100,subset_31,subset_128' '1220'
distillation 'subset_143,subset_220,subset_154,subset_129,subset_59,subset_55,subset_23,subset_7,subset_8,subset_108,subset_151,subset_22,subset_139,subset_219,subset_173,subset_26,subset_188,subset_35,subset_57,subset_62,subset_70,subset_189' 'subset_118,subset_73,subset_112,subset_119,subset_92,subset_16,subset_69,subset_27,subset_62,subset_85,subset_71,subset_60,subset_135' '60068'
# Epoch 2
distillation 'subset_7,subset_56,subset_214,subset_34,subset_69,subset_171,subset_58,subset_116,subset_217,subset_36,subset_173,subset_167,subset_175,subset_67,subset_28,subset_165,subset_49,subset_220,subset_98,subset_181,subset_14,subset_134' 'subset_70,subset_77,subset_37,subset_38,subset_69,subset_10,subset_91,subset_129,subset_68,subset_117,subset_115,subset_113,subset_12' '36471'
distillation 'subset_88,subset_219,subset_144,subset_163,subset_73,subset_204,subset_202,subset_99,subset_4,subset_161,subset_155,subset_25,subset_156,subset_176,subset_177,subset_3,subset_183,subset_40,subset_72,subset_110,subset_102,subset_101' 'subset_103,subset_130,subset_126,subset_1,subset_11,subset_99,subset_20,subset_136,subset_134,subset_9,subset_14,subset_25,subset_76' '98567'
distillation 'subset_189,subset_152,subset_141,subset_89,subset_178,subset_198,subset_39,subset_222,subset_55,subset_221,subset_168,subset_62,subset_79,subset_97,subset_112,subset_84,subset_200,subset_68,subset_124,subset_27,subset_1,subset_145' 'subset_28,subset_73,subset_31,subset_7,subset_74,subset_105,subset_80,subset_51,subset_93,subset_97,subset_57,subset_75,subset_49' '55255'
distillation 'subset_63,subset_0,subset_137,subset_94,subset_105,subset_93,subset_197,subset_126,subset_122,subset_148,subset_184,subset_23,subset_85,subset_151,subset_46,subset_61,subset_50,subset_120,subset_174,subset_128,subset_104,subset_135' 'subset_54,subset_101,subset_79,subset_94,subset_118,subset_67,subset_114,subset_46,subset_6,subset_71,subset_106,subset_60,subset_23' '33065'
distillation 'subset_96,subset_121,subset_95,subset_106,subset_196,subset_100,subset_109,subset_103,subset_48,subset_108,subset_186,subset_51,subset_52,subset_77,subset_192,subset_9,subset_215,subset_125,subset_2,subset_132,subset_107,subset_172' 'subset_19,subset_89,subset_44,subset_56,subset_78,subset_0,subset_45,subset_123,subset_64,subset_35,subset_5,subset_4,subset_107' '10735'
distillation 'subset_80,subset_118,subset_133,subset_193,subset_70,subset_6,subset_17,subset_129,subset_92,subset_45,subset_130,subset_10,subset_91,subset_26,subset_54,subset_16,subset_179,subset_43,subset_83,subset_59,subset_191,subset_195' 'subset_125,subset_58,subset_81,subset_47,subset_15,subset_131,subset_120,subset_119,subset_3,subset_32,subset_111,subset_41,subset_26' '61644'
distillation 'subset_149,subset_66,subset_20,subset_119,subset_71,subset_142,subset_131,subset_166,subset_209,subset_207,subset_199,subset_41,subset_115,subset_32,subset_212,subset_42,subset_164,subset_15,subset_158,subset_182,subset_139,subset_208' 'subset_82,subset_104,subset_65,subset_90,subset_135,subset_8,subset_127,subset_102,subset_27,subset_33,subset_13,subset_87,subset_48' '2540'
distillation 'subset_8,subset_216,subset_44,subset_30,subset_154,subset_53,subset_33,subset_150,subset_113,subset_81,subset_213,subset_75,subset_188,subset_76,subset_162,subset_159,subset_37,subset_157,subset_86,subset_29,subset_123,subset_64' 'subset_133,subset_21,subset_112,subset_95,subset_43,subset_137,subset_29,subset_62,subset_34,subset_96,subset_52,subset_61,subset_132' '98176'
distillation 'subset_201,subset_57,subset_210,subset_160,subset_24,subset_12,subset_153,subset_87,subset_38,subset_74,subset_22,subset_180,subset_190,subset_143,subset_114,subset_194,subset_127,subset_35,subset_5,subset_169,subset_117,subset_187' 'subset_16,subset_18,subset_22,subset_50,subset_98,subset_109,subset_36,subset_30,subset_72,subset_122,subset_124,subset_110,subset_84' '70702'
distillation 'subset_18,subset_11,subset_185,subset_211,subset_31,subset_170,subset_218,subset_203,subset_136,subset_82,subset_205,subset_140,subset_65,subset_78,subset_60,subset_19,subset_47,subset_111,subset_206,subset_138,subset_146,subset_13' 'subset_63,subset_2,subset_66,subset_86,subset_40,subset_116,subset_17,subset_24,subset_108,subset_100,subset_121,subset_39,subset_88' '6826'
# Epoch 3
distillation 'subset_149,subset_128,subset_199,subset_78,subset_198,subset_30,subset_17,subset_123,subset_215,subset_158,subset_19,subset_79,subset_9,subset_178,subset_146,subset_84,subset_67,subset_76,subset_101,subset_25,subset_85,subset_36' 'subset_106,subset_132,subset_6,subset_98,subset_92,subset_62,subset_84,subset_32,subset_114,subset_46,subset_12,subset_26,subset_131' '12217'
distillation 'subset_46,subset_1,subset_74,subset_38,subset_183,subset_14,subset_100,subset_221,subset_70,subset_163,subset_69,subset_119,subset_125,subset_49,subset_193,subset_81,subset_134,subset_186,subset_114,subset_138,subset_24,subset_131' 'subset_104,subset_125,subset_108,subset_68,subset_25,subset_124,subset_43,subset_116,subset_72,subset_3,subset_21,subset_60,subset_77' '29451'
distillation 'subset_2,subset_22,subset_103,subset_148,subset_31,subset_43,subset_98,subset_126,subset_150,subset_192,subset_213,subset_217,subset_61,subset_210,subset_7,subset_185,subset_52,subset_197,subset_28,subset_51,subset_212,subset_175' 'subset_103,subset_18,subset_11,subset_55,subset_71,subset_59,subset_66,subset_118,subset_29,subset_5,subset_1,subset_121,subset_0' '14871'
distillation 'subset_177,subset_153,subset_136,subset_205,subset_154,subset_195,subset_219,subset_32,subset_127,subset_72,subset_5,subset_165,subset_111,subset_82,subset_75,subset_156,subset_112,subset_139,subset_18,subset_180,subset_191,subset_189' 'subset_112,subset_120,subset_109,subset_50,subset_27,subset_34,subset_2,subset_42,subset_22,subset_89,subset_30,subset_91,subset_94' '60482'
distillation 'subset_0,subset_105,subset_57,subset_89,subset_122,subset_174,subset_8,subset_161,subset_107,subset_60,subset_202,subset_3,subset_167,subset_214,subset_12,subset_90,subset_91,subset_106,subset_37,subset_200,subset_162,subset_209' 'subset_53,subset_137,subset_28,subset_47,subset_74,subset_8,subset_105,subset_19,subset_85,subset_73,subset_100,subset_14,subset_97' '15396'
distillation 'subset_15,subset_130,subset_166,subset_55,subset_44,subset_203,subset_159,subset_144,subset_83,subset_56,subset_207,subset_20,subset_65,subset_147,subset_184,subset_133,subset_176,subset_113,subset_63,subset_95,subset_168,subset_54' 'subset_90,subset_58,subset_64,subset_9,subset_82,subset_102,subset_37,subset_39,subset_113,subset_38,subset_80,subset_79,subset_56' '84896'
distillation 'subset_141,subset_120,subset_179,subset_47,subset_104,subset_108,subset_92,subset_41,subset_6,subset_93,subset_88,subset_64,subset_50,subset_137,subset_86,subset_211,subset_26,subset_142,subset_94,subset_66,subset_152,subset_118' 'subset_10,subset_95,subset_115,subset_133,subset_4,subset_88,subset_107,subset_122,subset_23,subset_49,subset_13,subset_123,subset_126' '20181'
distillation 'subset_140,subset_129,subset_196,subset_80,subset_190,subset_16,subset_132,subset_34,subset_135,subset_59,subset_13,subset_187,subset_172,subset_71,subset_96,subset_206,subset_170,subset_4,subset_208,subset_23,subset_73,subset_124' 'subset_78,subset_40,subset_119,subset_41,subset_54,subset_134,subset_129,subset_63,subset_99,subset_93,subset_83,subset_135,subset_31' '65323'
distillation 'subset_33,subset_201,subset_182,subset_181,subset_39,subset_77,subset_48,subset_29,subset_42,subset_143,subset_58,subset_164,subset_204,subset_151,subset_110,subset_218,subset_87,subset_194,subset_169,subset_216,subset_102,subset_109' 'subset_15,subset_130,subset_44,subset_76,subset_61,subset_127,subset_111,subset_65,subset_52,subset_96,subset_51,subset_24,subset_33' '93926'
distillation 'subset_117,subset_145,subset_160,subset_45,subset_62,subset_115,subset_99,subset_157,subset_10,subset_155,subset_68,subset_40,subset_53,subset_171,subset_222,subset_27,subset_116,subset_35,subset_173,subset_21,subset_11,subset_220' 'subset_117,subset_45,subset_36,subset_87,subset_17,subset_48,subset_57,subset_7,subset_69,subset_75,subset_86,subset_110,subset_16' '38254'
# Epoch 4
distillation 'subset_88,subset_161,subset_47,subset_74,subset_90,subset_162,subset_115,subset_204,subset_183,subset_210,subset_212,subset_35,subset_170,subset_167,subset_12,subset_195,subset_50,subset_84,subset_89,subset_14,subset_119,subset_99' 'subset_55,subset_58,subset_75,subset_124,subset_51,subset_79,subset_95,subset_103,subset_126,subset_11,subset_7,subset_12,subset_136' '64260'
distillation 'subset_64,subset_208,subset_164,subset_198,subset_181,subset_23,subset_190,subset_10,subset_45,subset_145,subset_6,subset_58,subset_209,subset_126,subset_100,subset_194,subset_177,subset_17,subset_103,subset_215,subset_135,subset_67' 'subset_73,subset_34,subset_112,subset_118,subset_39,subset_77,subset_97,subset_94,subset_90,subset_101,subset_76,subset_98,subset_109' '93375'
distillation 'subset_34,subset_9,subset_114,subset_175,subset_200,subset_139,subset_146,subset_105,subset_155,subset_192,subset_187,subset_165,subset_113,subset_110,subset_83,subset_3,subset_131,subset_144,subset_216,subset_57,subset_49,subset_98' 'subset_102,subset_21,subset_48,subset_66,subset_28,subset_120,subset_93,subset_116,subset_37,subset_3,subset_115,subset_27,subset_132' '58138'
distillation 'subset_199,subset_117,subset_142,subset_56,subset_152,subset_15,subset_178,subset_201,subset_43,subset_68,subset_75,subset_174,subset_122,subset_213,subset_71,subset_72,subset_2,subset_61,subset_48,subset_137,subset_188,subset_211' 'subset_35,subset_56,subset_65,subset_29,subset_69,subset_8,subset_44,subset_106,subset_52,subset_46,subset_57,subset_72,subset_111' '9632'
distillation 'subset_40,subset_77,subset_128,subset_42,subset_8,subset_196,subset_59,subset_29,subset_73,subset_221,subset_129,subset_127,subset_51,subset_70,subset_141,subset_82,subset_182,subset_111,subset_19,subset_46,subset_36,subset_120' 'subset_13,subset_47,subset_121,subset_85,subset_113,subset_89,subset_134,subset_23,subset_104,subset_1,subset_38,subset_68,subset_59' '10589'
distillation 'subset_65,subset_160,subset_102,subset_150,subset_87,subset_163,subset_101,subset_217,subset_148,subset_22,subset_222,subset_123,subset_33,subset_138,subset_30,subset_140,subset_203,subset_169,subset_24,subset_55,subset_54,subset_80' 'subset_50,subset_16,subset_6,subset_107,subset_26,subset_15,subset_108,subset_60,subset_49,subset_91,subset_32,subset_2,subset_99' '42219'
distillation 'subset_130,subset_1,subset_7,subset_32,subset_52,subset_157,subset_13,subset_92,subset_184,subset_4,subset_206,subset_104,subset_132,subset_85,subset_214,subset_191,subset_153,subset_173,subset_25,subset_186,subset_91,subset_193' 'subset_135,subset_78,subset_42,subset_61,subset_40,subset_86,subset_20,subset_31,subset_96,subset_5,subset_117,subset_10,subset_100' '79730'
distillation 'subset_78,subset_97,subset_179,subset_16,subset_166,subset_38,subset_18,subset_66,subset_63,subset_69,subset_158,subset_176,subset_151,subset_172,subset_124,subset_60,subset_20,subset_156,subset_79,subset_106,subset_185,subset_171' 'subset_127,subset_45,subset_63,subset_43,subset_119,subset_83,subset_80,subset_137,subset_82,subset_70,subset_67,subset_122,subset_22' '19442'
distillation 'subset_86,subset_197,subset_53,subset_112,subset_116,subset_107,subset_41,subset_11,subset_180,subset_0,subset_21,subset_108,subset_44,subset_134,subset_133,subset_96,subset_26,subset_94,subset_5,subset_143,subset_219,subset_149' 'subset_105,subset_30,subset_24,subset_9,subset_71,subset_131,subset_17,subset_14,subset_36,subset_92,subset_62,subset_25,subset_125' '8609'
distillation 'subset_136,subset_95,subset_189,subset_220,subset_62,subset_109,subset_154,subset_76,subset_125,subset_218,subset_121,subset_118,subset_31,subset_202,subset_81,subset_93,subset_159,subset_28,subset_207,subset_147,subset_27,subset_37' 'subset_53,subset_110,subset_0,subset_123,subset_81,subset_128,subset_130,subset_133,subset_114,subset_4,subset_74,subset_88,subset_84' '16540'
# Epoch 5
distillation 'subset_3,subset_158,subset_210,subset_203,subset_122,subset_52,subset_49,subset_211,subset_129,subset_86,subset_45,subset_26,subset_155,subset_46,subset_51,subset_116,subset_50,subset_170,subset_179,subset_220,subset_214,subset_124' 'subset_28,subset_62,subset_21,subset_1,subset_38,subset_10,subset_58,subset_106,subset_107,subset_55,subset_93,subset_9,subset_30' '18030'
distillation 'subset_18,subset_80,subset_131,subset_140,subset_62,subset_76,subset_182,subset_156,subset_160,subset_199,subset_137,subset_70,subset_43,subset_89,subset_66,subset_148,subset_103,subset_42,subset_2,subset_82,subset_139,subset_132' 'subset_34,subset_18,subset_109,subset_47,subset_85,subset_129,subset_39,subset_25,subset_75,subset_108,subset_100,subset_136,subset_26' '20308'
distillation 'subset_130,subset_144,subset_47,subset_0,subset_218,subset_90,subset_64,subset_105,subset_135,subset_126,subset_92,subset_207,subset_9,subset_1,subset_106,subset_183,subset_33,subset_168,subset_150,subset_57,subset_39,subset_60' 'subset_54,subset_94,subset_120,subset_96,subset_66,subset_37,subset_68,subset_19,subset_117,subset_118,subset_13,subset_111,subset_122' '10114'
distillation 'subset_185,subset_194,subset_6,subset_24,subset_40,subset_177,subset_97,subset_88,subset_13,subset_146,subset_190,subset_162,subset_102,subset_222,subset_196,subset_81,subset_169,subset_189,subset_136,subset_34,subset_12,subset_11' 'subset_95,subset_29,subset_45,subset_46,subset_105,subset_0,subset_124,subset_22,subset_6,subset_90,subset_123,subset_64,subset_70' '38770'
distillation 'subset_22,subset_28,subset_217,subset_166,subset_174,subset_216,subset_159,subset_176,subset_113,subset_209,subset_133,subset_7,subset_69,subset_93,subset_73,subset_87,subset_14,subset_27,subset_212,subset_75,subset_178,subset_127' 'subset_77,subset_121,subset_76,subset_16,subset_119,subset_134,subset_116,subset_20,subset_48,subset_137,subset_112,subset_31,subset_101' '13256'
distillation 'subset_215,subset_8,subset_186,subset_175,subset_119,subset_67,subset_143,subset_192,subset_200,subset_153,subset_117,subset_79,subset_35,subset_74,subset_30,subset_72,subset_85,subset_54,subset_53,subset_96,subset_100,subset_157' 'subset_53,subset_7,subset_69,subset_73,subset_99,subset_127,subset_131,subset_86,subset_24,subset_92,subset_98,subset_49,subset_63' '66543'
distillation 'subset_161,subset_118,subset_94,subset_16,subset_56,subset_114,subset_204,subset_142,subset_115,subset_134,subset_37,subset_15,subset_154,subset_59,subset_95,subset_77,subset_23,subset_110,subset_138,subset_107,subset_187,subset_31' 'subset_79,subset_15,subset_128,subset_43,subset_57,subset_130,subset_2,subset_135,subset_12,subset_125,subset_114,subset_115,subset_52' '70748'
distillation 'subset_191,subset_181,subset_205,subset_213,subset_193,subset_201,subset_167,subset_36,subset_128,subset_5,subset_145,subset_48,subset_125,subset_104,subset_25,subset_195,subset_206,subset_58,subset_147,subset_44,subset_149,subset_197' 'subset_82,subset_51,subset_81,subset_74,subset_87,subset_11,subset_59,subset_88,subset_103,subset_32,subset_3,subset_67,subset_132' '96815'
distillation 'subset_99,subset_208,subset_84,subset_123,subset_171,subset_17,subset_29,subset_98,subset_20,subset_108,subset_120,subset_71,subset_164,subset_21,subset_19,subset_141,subset_202,subset_188,subset_111,subset_83,subset_163,subset_121' 'subset_14,subset_71,subset_41,subset_126,subset_113,subset_84,subset_56,subset_42,subset_17,subset_65,subset_78,subset_61,subset_40' '69013'
distillation 'subset_165,subset_180,subset_4,subset_219,subset_151,subset_172,subset_10,subset_112,subset_109,subset_198,subset_65,subset_78,subset_55,subset_101,subset_68,subset_32,subset_91,subset_221,subset_61,subset_41,subset_184,subset_173' 'subset_133,subset_36,subset_110,subset_5,subset_89,subset_91,subset_80,subset_60,subset_83,subset_23,subset_4,subset_44,subset_97' '4947'
# Epoch 6
distillation 'subset_22,subset_131,subset_55,subset_206,subset_35,subset_203,subset_172,subset_110,subset_30,subset_115,subset_173,subset_41,subset_96,subset_125,subset_204,subset_169,subset_83,subset_149,subset_136,subset_182,subset_104,subset_54' 'subset_136,subset_46,subset_115,subset_45,subset_68,subset_91,subset_111,subset_49,subset_112,subset_17,subset_35,subset_122,subset_96' '58979'
distillation 'subset_77,subset_221,subset_19,subset_170,subset_185,subset_50,subset_94,subset_207,subset_152,subset_184,subset_189,subset_165,subset_155,subset_174,subset_64,subset_130,subset_122,subset_197,subset_13,subset_87,subset_114,subset_156' 'subset_128,subset_62,subset_21,subset_94,subset_89,subset_131,subset_127,subset_13,subset_92,subset_19,subset_28,subset_109,subset_105' '61057'
distillation 'subset_132,subset_10,subset_36,subset_135,subset_43,subset_150,subset_37,subset_69,subset_128,subset_218,subset_29,subset_217,subset_154,subset_127,subset_2,subset_65,subset_14,subset_215,subset_18,subset_11,subset_38,subset_5' 'subset_90,subset_121,subset_26,subset_58,subset_7,subset_4,subset_104,subset_132,subset_93,subset_14,subset_77,subset_117,subset_137' '89143'
distillation 'subset_72,subset_0,subset_177,subset_52,subset_59,subset_71,subset_171,subset_79,subset_146,subset_147,subset_167,subset_61,subset_73,subset_32,subset_192,subset_45,subset_181,subset_101,subset_120,subset_153,subset_144,subset_102' 'subset_2,subset_108,subset_88,subset_40,subset_5,subset_9,subset_78,subset_44,subset_52,subset_48,subset_30,subset_73,subset_36' '67811'
distillation 'subset_3,subset_143,subset_111,subset_179,subset_166,subset_178,subset_118,subset_60,subset_196,subset_85,subset_6,subset_190,subset_57,subset_31,subset_129,subset_106,subset_21,subset_23,subset_142,subset_126,subset_98,subset_107' 'subset_11,subset_23,subset_34,subset_120,subset_65,subset_20,subset_101,subset_80,subset_126,subset_99,subset_116,subset_71,subset_31' '45156'
distillation 'subset_116,subset_103,subset_160,subset_66,subset_7,subset_51,subset_92,subset_194,subset_162,subset_151,subset_100,subset_40,subset_220,subset_138,subset_76,subset_137,subset_210,subset_15,subset_158,subset_188,subset_48,subset_84' 'subset_100,subset_59,subset_18,subset_37,subset_33,subset_29,subset_64,subset_74,subset_69,subset_85,subset_24,subset_41,subset_61' '16821'
distillation 'subset_164,subset_88,subset_86,subset_108,subset_213,subset_34,subset_140,subset_81,subset_211,subset_1,subset_4,subset_214,subset_133,subset_180,subset_25,subset_123,subset_93,subset_58,subset_212,subset_49,subset_28,subset_78' 'subset_15,subset_6,subset_133,subset_39,subset_32,subset_51,subset_86,subset_123,subset_125,subset_67,subset_8,subset_66,subset_107' '72260'
distillation 'subset_9,subset_80,subset_8,subset_148,subset_157,subset_67,subset_24,subset_99,subset_124,subset_17,subset_199,subset_161,subset_62,subset_134,subset_176,subset_74,subset_145,subset_46,subset_39,subset_89,subset_191,subset_105' 'subset_47,subset_38,subset_110,subset_103,subset_60,subset_129,subset_1,subset_97,subset_12,subset_87,subset_53,subset_50,subset_70' '83859'
distillation 'subset_91,subset_75,subset_16,subset_53,subset_209,subset_63,subset_200,subset_112,subset_95,subset_219,subset_159,subset_90,subset_117,subset_222,subset_168,subset_56,subset_183,subset_119,subset_141,subset_202,subset_97,subset_205' 'subset_84,subset_135,subset_55,subset_106,subset_3,subset_43,subset_118,subset_16,subset_79,subset_124,subset_42,subset_81,subset_10' '77042'
distillation 'subset_175,subset_33,subset_109,subset_44,subset_216,subset_27,subset_26,subset_186,subset_20,subset_139,subset_198,subset_201,subset_121,subset_208,subset_68,subset_47,subset_82,subset_42,subset_113,subset_187,subset_70,subset_193' 'subset_83,subset_0,subset_119,subset_114,subset_27,subset_76,subset_82,subset_63,subset_134,subset_95,subset_72,subset_130,subset_102' '23850'
# Epoch 7
distillation 'subset_190,subset_70,subset_66,subset_111,subset_34,subset_89,subset_137,subset_155,subset_134,subset_222,subset_130,subset_120,subset_58,subset_35,subset_159,subset_153,subset_206,subset_40,subset_141,subset_22,subset_194,subset_126' 'subset_20,subset_115,subset_121,subset_81,subset_63,subset_122,subset_28,subset_6,subset_10,subset_84,subset_120,subset_105,subset_97' '24420'
distillation 'subset_187,subset_77,subset_191,subset_179,subset_94,subset_185,subset_178,subset_212,subset_8,subset_182,subset_117,subset_209,subset_154,subset_78,subset_132,subset_29,subset_44,subset_76,subset_207,subset_121,subset_57,subset_72' 'subset_36,subset_29,subset_46,subset_41,subset_35,subset_17,subset_50,subset_58,subset_24,subset_13,subset_114,subset_80,subset_44' '98173'
distillation 'subset_30,subset_147,subset_135,subset_23,subset_31,subset_143,subset_175,subset_122,subset_188,subset_52,subset_43,subset_107,subset_220,subset_54,subset_95,subset_119,subset_170,subset_202,subset_161,subset_83,subset_80,subset_174' 'subset_136,subset_130,subset_123,subset_104,subset_25,subset_134,subset_92,subset_61,subset_34,subset_129,subset_40,subset_62,subset_66' '37941'
distillation 'subset_198,subset_214,subset_150,subset_115,subset_201,subset_33,subset_32,subset_7,subset_71,subset_186,subset_60,subset_46,subset_203,subset_140,subset_69,subset_36,subset_4,subset_149,subset_217,subset_125,subset_131,subset_86' 'subset_45,subset_65,subset_60,subset_119,subset_2,subset_111,subset_57,subset_30,subset_42,subset_74,subset_43,subset_93,subset_96' '4910'
distillation 'subset_109,subset_14,subset_200,subset_101,subset_74,subset_28,subset_211,subset_196,subset_172,subset_10,subset_183,subset_91,subset_152,subset_2,subset_27,subset_216,subset_75,subset_55,subset_98,subset_90,subset_103,subset_65' 'subset_128,subset_7,subset_52,subset_94,subset_135,subset_133,subset_0,subset_71,subset_124,subset_85,subset_100,subset_91,subset_22' '1214'
distillation 'subset_5,subset_20,subset_100,subset_85,subset_48,subset_112,subset_166,subset_79,subset_208,subset_21,subset_16,subset_88,subset_210,subset_108,subset_39,subset_73,subset_163,subset_6,subset_219,subset_96,subset_19,subset_105' 'subset_131,subset_69,subset_73,subset_53,subset_21,subset_47,subset_51,subset_118,subset_9,subset_72,subset_87,subset_98,subset_106' '39110'
distillation 'subset_11,subset_51,subset_177,subset_113,subset_151,subset_215,subset_180,subset_50,subset_84,subset_118,subset_12,subset_129,subset_139,subset_45,subset_192,subset_64,subset_102,subset_128,subset_162,subset_99,subset_145,subset_160' 'subset_77,subset_39,subset_54,subset_137,subset_55,subset_14,subset_4,subset_109,subset_70,subset_83,subset_132,subset_75,subset_32' '74443'
distillation 'subset_13,subset_169,subset_136,subset_92,subset_167,subset_106,subset_18,subset_138,subset_195,subset_146,subset_81,subset_205,subset_15,subset_53,subset_17,subset_157,subset_142,subset_25,subset_218,subset_62,subset_0,subset_123' 'subset_67,subset_38,subset_125,subset_78,subset_33,subset_113,subset_37,subset_101,subset_19,subset_95,subset_102,subset_18,subset_99' '79007'
distillation 'subset_56,subset_41,subset_26,subset_110,subset_173,subset_133,subset_144,subset_9,subset_67,subset_176,subset_168,subset_82,subset_93,subset_49,subset_156,subset_199,subset_193,subset_221,subset_213,subset_148,subset_47,subset_24' 'subset_86,subset_88,subset_64,subset_27,subset_59,subset_89,subset_107,subset_49,subset_110,subset_103,subset_3,subset_16,subset_126' '14064'
distillation 'subset_189,subset_59,subset_1,subset_124,subset_37,subset_38,subset_104,subset_171,subset_42,subset_3,subset_127,subset_63,subset_181,subset_87,subset_114,subset_61,subset_197,subset_164,subset_184,subset_68,subset_204,subset_158' 'subset_15,subset_12,subset_31,subset_82,subset_90,subset_8,subset_11,subset_127,subset_23,subset_108,subset_79,subset_68,subset_48' '43914'
# Epoch 8
distillation 'subset_158,subset_209,subset_29,subset_169,subset_129,subset_182,subset_139,subset_42,subset_118,subset_18,subset_10,subset_125,subset_126,subset_200,subset_208,subset_67,subset_179,subset_24,subset_141,subset_127,subset_142,subset_68' 'subset_54,subset_0,subset_13,subset_21,subset_81,subset_3,subset_95,subset_94,subset_2,subset_22,subset_127,subset_20,subset_39' '8908'
distillation 'subset_110,subset_213,subset_133,subset_13,subset_75,subset_19,subset_147,subset_106,subset_69,subset_1,subset_124,subset_80,subset_90,subset_137,subset_6,subset_35,subset_73,subset_202,subset_2,subset_88,subset_92,subset_91' 'subset_44,subset_19,subset_75,subset_63,subset_50,subset_101,subset_56,subset_91,subset_4,subset_90,subset_52,subset_72,subset_10' '92690'
distillation 'subset_176,subset_58,subset_201,subset_152,subset_214,subset_85,subset_47,subset_108,subset_36,subset_38,subset_39,subset_160,subset_162,subset_135,subset_72,subset_212,subset_180,subset_41,subset_7,subset_172,subset_51,subset_5' 'subset_102,subset_125,subset_27,subset_8,subset_69,subset_109,subset_48,subset_131,subset_24,subset_5,subset_111,subset_15,subset_49' '19795'
distillation 'subset_123,subset_168,subset_11,subset_14,subset_193,subset_157,subset_130,subset_28,subset_60,subset_12,subset_31,subset_144,subset_103,subset_40,subset_77,subset_188,subset_132,subset_34,subset_116,subset_109,subset_55,subset_61' 'subset_65,subset_62,subset_129,subset_41,subset_77,subset_83,subset_29,subset_1,subset_121,subset_118,subset_110,subset_100,subset_36' '1198'
distillation 'subset_134,subset_148,subset_43,subset_89,subset_222,subset_138,subset_185,subset_17,subset_190,subset_117,subset_175,subset_53,subset_151,subset_102,subset_59,subset_119,subset_33,subset_100,subset_97,subset_20,subset_52,subset_64' 'subset_58,subset_31,subset_47,subset_30,subset_96,subset_78,subset_104,subset_9,subset_135,subset_88,subset_114,subset_61,subset_45' '28668'
distillation 'subset_206,subset_95,subset_155,subset_167,subset_54,subset_104,subset_112,subset_120,subset_161,subset_71,subset_101,subset_94,subset_86,subset_159,subset_99,subset_15,subset_49,subset_87,subset_57,subset_44,subset_122,subset_164' 'subset_57,subset_16,subset_76,subset_40,subset_60,subset_11,subset_93,subset_117,subset_116,subset_18,subset_46,subset_67,subset_14' '66287'
distillation 'subset_199,subset_181,subset_22,subset_171,subset_113,subset_62,subset_204,subset_194,subset_26,subset_107,subset_215,subset_16,subset_84,subset_98,subset_196,subset_165,subset_76,subset_173,subset_184,subset_145,subset_220,subset_211' 'subset_115,subset_82,subset_126,subset_112,subset_85,subset_130,subset_107,subset_87,subset_33,subset_84,subset_122,subset_12,subset_106' '59770'
distillation 'subset_189,subset_195,subset_140,subset_219,subset_156,subset_207,subset_203,subset_197,subset_8,subset_153,subset_0,subset_74,subset_56,subset_66,subset_192,subset_128,subset_46,subset_63,subset_150,subset_48,subset_170,subset_183' 'subset_70,subset_74,subset_34,subset_28,subset_80,subset_38,subset_23,subset_105,subset_79,subset_133,subset_120,subset_55,subset_119' '48916'
distillation 'subset_198,subset_70,subset_83,subset_4,subset_210,subset_78,subset_37,subset_191,subset_25,subset_27,subset_187,subset_149,subset_105,subset_65,subset_216,subset_146,subset_136,subset_30,subset_114,subset_221,subset_218,subset_45' 'subset_136,subset_92,subset_134,subset_43,subset_35,subset_73,subset_103,subset_7,subset_128,subset_51,subset_6,subset_71,subset_99' '7923'
distillation 'subset_186,subset_131,subset_93,subset_111,subset_23,subset_9,subset_174,subset_205,subset_79,subset_82,subset_121,subset_217,subset_178,subset_3,subset_143,subset_115,subset_21,subset_96,subset_154,subset_163,subset_177,subset_50' 'subset_17,subset_32,subset_89,subset_26,subset_123,subset_59,subset_25,subset_124,subset_53,subset_68,subset_42,subset_113,subset_98' '80980'


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


