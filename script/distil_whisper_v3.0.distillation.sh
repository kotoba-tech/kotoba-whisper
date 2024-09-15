export WER_THRESHOLD=10.0  # WER threshold applied at data filtering.
export TEACHER_MODEL="openai/whisper-large-v3"  # Teacher model for the distillation.
export HF_ORG="japanese-asr"  # HuggingFace organization to push the artifacts
export HF_MODEL_ALIAS="distil-whisper-bilingual"  # Model alias used when pushing models.


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
    --dataset_config_name_2 "${MODEL_CONFIG_2}" \
    --dataset_feature_2 "whisper_transcription,transcription/ja_gpt3.5" \
    --dataset_language_2 "en,ja" \
    --dataset_task_2 "transcribe,translate" \
    --dataset_timestamp_2 "true,false" \
    --dataset_kl_2 "true,false" \
    --max_label_length 128 \
    --learning_rate 0.0001 \
    --logging_steps 25 \
    --attn_implementation "flash_attention_2" \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --output_dir "./${HF_MODEL_ALIAS}" \
    --overwrite_output_dir \
    --seed ${SEED} \
    --report_to "none" \
    --num_train_epochs 1
}
# Epoch 1
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_159,subset_110,subset_123,subset_121,subset_98,subset_181,subset_60,subset_13,subset_44,subset_16,subset_84' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_113,subset_4,subset_26,subset_131,subset_38' &
distillation 'subset_162,subset_36,subset_185,subset_2,subset_45,subset_105,subset_195,subset_168,subset_52,subset_208,subset_212' 'subset_67,subset_121,subset_126,subset_78,subset_134,subset_14' '88039'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_135,subset_29,subset_215,subset_21,subset_169,subset_194,subset_30,subset_127,subset_175,subset_4,subset_42' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_74,subset_40,subset_75,subset_98,subset_117,subset_22' &
distillation 'subset_159,subset_110,subset_123,subset_121,subset_98,subset_181,subset_60,subset_13,subset_44,subset_16,subset_84' 'subset_72,subset_113,subset_4,subset_26,subset_131,subset_38' '84607'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_47,subset_5,subset_77,subset_3,subset_15,subset_216,subset_190,subset_205,subset_81,subset_192,subset_99' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_53,subset_47,subset_48,subset_64,subset_63,subset_25' &
distillation 'subset_135,subset_29,subset_215,subset_21,subset_169,subset_194,subset_30,subset_127,subset_175,subset_4,subset_42' 'subset_74,subset_40,subset_75,subset_98,subset_117,subset_22' '39321'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_133,subset_66,subset_145,subset_73,subset_10,subset_122,subset_177,subset_9,subset_12,subset_38,subset_138' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_19,subset_28,subset_5,subset_107,subset_2,subset_81' &
distillation 'subset_47,subset_5,subset_77,subset_3,subset_15,subset_216,subset_190,subset_205,subset_81,subset_192,subset_99' 'subset_53,subset_47,subset_48,subset_64,subset_63,subset_25' '59929'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_61,subset_198,subset_89,subset_193,subset_134,subset_119,subset_79,subset_78,subset_209,subset_64,subset_103' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_65,subset_36,subset_10,subset_33,subset_32,subset_127' &
distillation 'subset_133,subset_66,subset_145,subset_73,subset_10,subset_122,subset_177,subset_9,subset_12,subset_38,subset_138' 'subset_19,subset_28,subset_5,subset_107,subset_2,subset_81' '41441'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_0,subset_111,subset_196,subset_180,subset_37,subset_222,subset_140,subset_126,subset_104,subset_144,subset_187' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_41,subset_39,subset_20,subset_49,subset_34,subset_37' &
distillation 'subset_61,subset_198,subset_89,subset_193,subset_134,subset_119,subset_79,subset_78,subset_209,subset_64,subset_103' 'subset_65,subset_36,subset_10,subset_33,subset_32,subset_127' '98548'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_218,subset_32,subset_171,subset_152,subset_85,subset_48,subset_161,subset_125,subset_76,subset_100,subset_153' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_91,subset_106,subset_76,subset_79,subset_80' &
distillation 'subset_0,subset_111,subset_196,subset_180,subset_37,subset_222,subset_140,subset_126,subset_104,subset_144,subset_187' 'subset_41,subset_39,subset_20,subset_49,subset_34,subset_37' '9508'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_128,subset_19,subset_131,subset_199,subset_156,subset_115,subset_65,subset_184,subset_221,subset_46,subset_51' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_105,subset_42,subset_97,subset_99,subset_46,subset_44' &
distillation 'subset_218,subset_32,subset_171,subset_152,subset_85,subset_48,subset_161,subset_125,subset_76,subset_100,subset_153' 'subset_3,subset_91,subset_106,subset_76,subset_79,subset_80' '1220'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_101,subset_54,subset_172,subset_106,subset_109,subset_142,subset_149,subset_95,subset_167,subset_157,subset_33' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_66,subset_129,subset_29,subset_133,subset_50,subset_122' &
distillation 'subset_128,subset_19,subset_131,subset_199,subset_156,subset_115,subset_65,subset_184,subset_221,subset_46,subset_51' 'subset_105,subset_42,subset_97,subset_99,subset_46,subset_44' '60068'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_164,subset_130,subset_82,subset_113,subset_197,subset_63,subset_132,subset_124,subset_186,subset_174,subset_112' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_87,subset_0,subset_88,subset_77,subset_95,subset_94' &
distillation 'subset_101,subset_54,subset_172,subset_106,subset_109,subset_142,subset_149,subset_95,subset_167,subset_157,subset_33' 'subset_66,subset_129,subset_29,subset_133,subset_50,subset_122' '81416'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_72,subset_120,subset_148,subset_27,subset_201,subset_34,subset_102,subset_80,subset_206,subset_155,subset_14' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_124,subset_61,subset_125,subset_120,subset_21,subset_30' &
distillation 'subset_164,subset_130,subset_82,subset_113,subset_197,subset_63,subset_132,subset_124,subset_186,subset_174,subset_112' 'subset_87,subset_0,subset_88,subset_77,subset_95,subset_94' '73792'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_83,subset_191,subset_69,subset_211,subset_118,subset_146,subset_217,subset_136,subset_43,subset_18,subset_68' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_11,subset_136,subset_123,subset_114,subset_96,subset_70' &
distillation 'subset_72,subset_120,subset_148,subset_27,subset_201,subset_34,subset_102,subset_80,subset_206,subset_155,subset_14' 'subset_124,subset_61,subset_125,subset_120,subset_21,subset_30' '13104'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_53,subset_90,subset_94,subset_41,subset_93,subset_116,subset_182,subset_176,subset_25,subset_202,subset_165' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_137,subset_9,subset_90,subset_102,subset_35,subset_23' &
distillation 'subset_83,subset_191,subset_69,subset_211,subset_118,subset_146,subset_217,subset_136,subset_43,subset_18,subset_68' 'subset_11,subset_136,subset_123,subset_114,subset_96,subset_70' '9602'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_74,subset_58,subset_170,subset_17,subset_49,subset_147,subset_92,subset_158,subset_160,subset_75,subset_141' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_104,subset_132,subset_57,subset_68,subset_89,subset_110' &
distillation 'subset_53,subset_90,subset_94,subset_41,subset_93,subset_116,subset_182,subset_176,subset_25,subset_202,subset_165' 'subset_137,subset_9,subset_90,subset_102,subset_35,subset_23' '70468'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_20,subset_96,subset_31,subset_137,subset_117,subset_11,subset_67,subset_200,subset_88,subset_91,subset_24' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_115,subset_13,subset_43,subset_101,subset_51,subset_7' &
distillation 'subset_74,subset_58,subset_170,subset_17,subset_49,subset_147,subset_92,subset_158,subset_160,subset_75,subset_141' 'subset_104,subset_132,subset_57,subset_68,subset_89,subset_110' '27938'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_97,subset_204,subset_213,subset_86,subset_203,subset_39,subset_214,subset_87,subset_207,subset_178,subset_40' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_108,subset_82,subset_83,subset_86,subset_6,subset_93' &
distillation 'subset_20,subset_96,subset_31,subset_137,subset_117,subset_11,subset_67,subset_200,subset_88,subset_91,subset_24' 'subset_115,subset_13,subset_43,subset_101,subset_51,subset_7' '66307'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_1,subset_71,subset_150,subset_114,subset_56,subset_107,subset_210,subset_179,subset_166,subset_183,subset_50' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_59,subset_111,subset_54,subset_45,subset_55,subset_130' &
distillation 'subset_97,subset_204,subset_213,subset_86,subset_203,subset_39,subset_214,subset_87,subset_207,subset_178,subset_40' 'subset_108,subset_82,subset_83,subset_86,subset_6,subset_93' '34760'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_143,subset_220,subset_154,subset_129,subset_59,subset_55,subset_23,subset_7,subset_8,subset_108,subset_151' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_12,subset_24,subset_52,subset_103,subset_116' &
distillation 'subset_1,subset_71,subset_150,subset_114,subset_56,subset_107,subset_210,subset_179,subset_166,subset_183,subset_50' 'subset_59,subset_111,subset_54,subset_45,subset_55,subset_130' '17361'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_22,subset_139,subset_219,subset_173,subset_26,subset_188,subset_35,subset_57,subset_62,subset_70,subset_189' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_100,subset_31,subset_128,subset_118,subset_73,subset_112' &
distillation 'subset_143,subset_220,subset_154,subset_129,subset_59,subset_55,subset_23,subset_7,subset_8,subset_108,subset_151' 'subset_109,subset_12,subset_24,subset_52,subset_103,subset_116' '45745'
distillation 'subset_22,subset_139,subset_219,subset_173,subset_26,subset_188,subset_35,subset_57,subset_62,subset_70,subset_189' 'subset_100,subset_31,subset_128,subset_118,subset_73,subset_112' '45745'
# Epoch 2
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_171,subset_69,subset_221,subset_163,subset_109,subset_39,subset_148,subset_89,subset_197,subset_147,subset_59' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_37,subset_86,subset_53,subset_110,subset_116,subset_58' &
distillation 'subset_22,subset_139,subset_219,subset_173,subset_26,subset_188,subset_35,subset_57,subset_62,subset_70,subset_189' 'subset_100,subset_31,subset_128,subset_118,subset_73,subset_112' '2671'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_144,subset_176,subset_138,subset_101,subset_36,subset_146,subset_202,subset_217,subset_102,subset_80,subset_168' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_82,subset_78,subset_97,subset_38,subset_98,subset_84' &
distillation 'subset_171,subset_69,subset_221,subset_163,subset_109,subset_39,subset_148,subset_89,subset_197,subset_147,subset_59' 'subset_37,subset_86,subset_53,subset_110,subset_116,subset_58' '81439'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_166,subset_4,subset_79,subset_177,subset_92,subset_181,subset_178,subset_96,subset_84,subset_25,subset_175' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_24,subset_129,subset_117,subset_79,subset_40,subset_115' &
distillation 'subset_144,subset_176,subset_138,subset_101,subset_36,subset_146,subset_202,subset_217,subset_102,subset_80,subset_168' 'subset_82,subset_78,subset_97,subset_38,subset_98,subset_84' '19973'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_152,subset_179,subset_161,subset_132,subset_27,subset_16,subset_200,subset_17,subset_14,subset_156,subset_68' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_123,subset_1,subset_25,subset_2,subset_72,subset_49' &
distillation 'subset_166,subset_4,subset_79,subset_177,subset_92,subset_181,subset_178,subset_96,subset_84,subset_25,subset_175' 'subset_24,subset_129,subset_117,subset_79,subset_40,subset_115' '31266'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_10,subset_106,subset_134,subset_28,subset_141,subset_3,subset_118,subset_204,subset_73,subset_99,subset_85' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_8,subset_39,subset_31,subset_10,subset_114,subset_106' &
distillation 'subset_152,subset_179,subset_161,subset_132,subset_27,subset_16,subset_200,subset_17,subset_14,subset_156,subset_68' 'subset_123,subset_1,subset_25,subset_2,subset_72,subset_49' '16544'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_213,subset_93,subset_110,subset_222,subset_167,subset_13,subset_19,subset_184,subset_49,subset_150,subset_72' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_134,subset_15,subset_14,subset_51,subset_57,subset_91' &
distillation 'subset_10,subset_106,subset_134,subset_28,subset_141,subset_3,subset_118,subset_204,subset_73,subset_99,subset_85' 'subset_8,subset_39,subset_31,subset_10,subset_114,subset_106' '62070'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_56,subset_88,subset_98,subset_128,subset_54,subset_137,subset_165,subset_145,subset_34,subset_174,subset_0' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_99,subset_70,subset_93,subset_104,subset_28,subset_64' &
distillation 'subset_213,subset_93,subset_110,subset_222,subset_167,subset_13,subset_19,subset_184,subset_49,subset_150,subset_72' 'subset_134,subset_15,subset_14,subset_51,subset_57,subset_91' '87747'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_97,subset_143,subset_105,subset_58,subset_108,subset_126,subset_122,subset_86,subset_207,subset_57,subset_41' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_50,subset_121,subset_11,subset_60,subset_23,subset_69' &
distillation 'subset_56,subset_88,subset_98,subset_128,subset_54,subset_137,subset_165,subset_145,subset_34,subset_174,subset_0' 'subset_99,subset_70,subset_93,subset_104,subset_28,subset_64' '14993'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_83,subset_95,subset_61,subset_7,subset_214,subset_45,subset_155,subset_104,subset_135,subset_91,subset_121' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_85,subset_75,subset_20,subset_45,subset_101,subset_80' &
distillation 'subset_97,subset_143,subset_105,subset_58,subset_108,subset_126,subset_122,subset_86,subset_207,subset_57,subset_41' 'subset_50,subset_121,subset_11,subset_60,subset_23,subset_69' '73920'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_63,subset_186,subset_191,subset_100,subset_62,subset_103,subset_48,subset_120,subset_173,subset_51,subset_52' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_89,subset_136,subset_125,subset_103,subset_19,subset_94' &
distillation 'subset_83,subset_95,subset_61,subset_7,subset_214,subset_45,subset_155,subset_104,subset_135,subset_91,subset_121' 'subset_85,subset_75,subset_20,subset_45,subset_101,subset_80' '28569'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_77,subset_192,subset_9,subset_215,subset_125,subset_2,subset_20,subset_154,subset_130,subset_172,subset_22' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_9,subset_63,subset_52,subset_67,subset_0,subset_105' &
distillation 'subset_63,subset_186,subset_191,subset_100,subset_62,subset_103,subset_48,subset_120,subset_173,subset_51,subset_52' 'subset_89,subset_136,subset_125,subset_103,subset_19,subset_94' '60952'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_94,subset_133,subset_193,subset_124,subset_90,subset_21,subset_198,subset_112,subset_158,subset_46,subset_220' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_126,subset_111,subset_16,subset_5,subset_113,subset_92' &
distillation 'subset_77,subset_192,subset_9,subset_215,subset_125,subset_2,subset_20,subset_154,subset_130,subset_172,subset_22' 'subset_9,subset_63,subset_52,subset_67,subset_0,subset_105' '91680'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_70,subset_196,subset_116,subset_188,subset_71,subset_43,subset_195,subset_149,subset_66,subset_67,subset_119' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_62,subset_27,subset_68,subset_46,subset_44,subset_4' &
distillation 'subset_94,subset_133,subset_193,subset_124,subset_90,subset_21,subset_198,subset_112,subset_158,subset_46,subset_220' 'subset_126,subset_111,subset_16,subset_5,subset_113,subset_92' '33586'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_183,subset_142,subset_139,subset_131,subset_55,subset_209,subset_107,subset_199,subset_210,subset_115,subset_32' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_56,subset_35,subset_42,subset_81,subset_47,subset_22' &
distillation 'subset_70,subset_196,subset_116,subset_188,subset_71,subset_43,subset_195,subset_149,subset_66,subset_67,subset_119' 'subset_62,subset_27,subset_68,subset_46,subset_44,subset_4' '48351'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_212,subset_42,subset_164,subset_15,subset_23,subset_182,subset_6,subset_208,subset_35,subset_216,subset_44' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_131,subset_120,subset_119,subset_3,subset_71,subset_130' &
distillation 'subset_183,subset_142,subset_139,subset_131,subset_55,subset_209,subset_107,subset_199,subset_210,subset_115,subset_32' 'subset_56,subset_35,subset_42,subset_81,subset_47,subset_22' '21992'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_30,subset_151,subset_53,subset_33,subset_50,subset_113,subset_81,subset_40,subset_75,subset_47,subset_76' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_41,subset_13,subset_26,subset_118,subset_6,subset_65' &
distillation 'subset_212,subset_42,subset_164,subset_15,subset_23,subset_182,subset_6,subset_208,subset_35,subset_216,subset_44' 'subset_131,subset_120,subset_119,subset_3,subset_71,subset_130' '79415'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_162,subset_159,subset_37,subset_157,subset_1,subset_29,subset_123,subset_64,subset_201,subset_206,subset_129' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_90,subset_88,subset_124,subset_83,subset_127,subset_102' &
distillation 'subset_30,subset_151,subset_53,subset_33,subset_50,subset_113,subset_81,subset_40,subset_75,subset_47,subset_76' 'subset_41,subset_13,subset_26,subset_118,subset_6,subset_65' '79593'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_160,subset_24,subset_12,subset_153,subset_87,subset_38,subset_74,subset_189,subset_180,subset_190,subset_219' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_135,subset_112,subset_33,subset_12,subset_87,subset_7' &
distillation 'subset_162,subset_159,subset_37,subset_157,subset_1,subset_29,subset_123,subset_64,subset_201,subset_206,subset_129' 'subset_90,subset_88,subset_124,subset_83,subset_127,subset_102' '98032'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_114,subset_194,subset_127,subset_111,subset_5,subset_169,subset_117,subset_187,subset_18,subset_11,subset_185' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_48,subset_133,subset_21,subset_55,subset_66,subset_95' &
distillation 'subset_160,subset_24,subset_12,subset_153,subset_87,subset_38,subset_74,subset_189,subset_180,subset_190,subset_219' 'subset_135,subset_112,subset_33,subset_12,subset_87,subset_7' '94156'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_211,subset_31,subset_8,subset_170,subset_218,subset_203,subset_136,subset_26,subset_82,subset_205,subset_140' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_43,subset_54,subset_137,subset_18,subset_100,subset_108' &
distillation 'subset_114,subset_194,subset_127,subset_111,subset_5,subset_169,subset_117,subset_187,subset_18,subset_11,subset_185' 'subset_48,subset_133,subset_21,subset_55,subset_66,subset_95' '15012'
distillation 'subset_211,subset_31,subset_8,subset_170,subset_218,subset_203,subset_136,subset_26,subset_82,subset_205,subset_140' 'subset_43,subset_54,subset_137,subset_18,subset_100,subset_108' '15012'
# Epoch 3
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_212,subset_108,subset_180,subset_201,subset_167,subset_165,subset_38,subset_40,subset_115,subset_145,subset_110' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_23,subset_2,subset_118,subset_34,subset_8,subset_30' &
distillation 'subset_211,subset_31,subset_8,subset_170,subset_218,subset_203,subset_136,subset_26,subset_82,subset_205,subset_140' 'subset_43,subset_54,subset_137,subset_18,subset_100,subset_108' '50277'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_12,subset_195,subset_171,subset_65,subset_58,subset_218,subset_3,subset_9,subset_67,subset_197,subset_99' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_127,subset_82,subset_60,subset_120,subset_21,subset_17' &
distillation 'subset_212,subset_108,subset_180,subset_201,subset_167,subset_165,subset_38,subset_40,subset_115,subset_145,subset_110' 'subset_23,subset_2,subset_118,subset_34,subset_8,subset_30' '24983'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_80,subset_47,subset_22,subset_63,subset_20,subset_143,subset_120,subset_217,subset_53,subset_45,subset_149' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_87,subset_86,subset_117,subset_121,subset_85' &
distillation 'subset_12,subset_195,subset_171,subset_65,subset_58,subset_218,subset_3,subset_9,subset_67,subset_197,subset_99' 'subset_127,subset_82,subset_60,subset_120,subset_21,subset_17' '78561'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_51,subset_178,subset_207,subset_152,subset_101,subset_169,subset_170,subset_150,subset_186,subset_203,subset_185' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_26,subset_43,subset_10,subset_134,subset_70,subset_38' &
distillation 'subset_80,subset_47,subset_22,subset_63,subset_20,subset_143,subset_120,subset_217,subset_53,subset_45,subset_149' 'subset_109,subset_87,subset_86,subset_117,subset_121,subset_85' '66616'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_78,subset_44,subset_147,subset_131,subset_49,subset_8,subset_29,subset_187,subset_107,subset_177,subset_46' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_56,subset_3,subset_89,subset_131,subset_1,subset_137' &
distillation 'subset_51,subset_178,subset_207,subset_152,subset_101,subset_169,subset_170,subset_150,subset_186,subset_203,subset_185' 'subset_26,subset_43,subset_10,subset_134,subset_70,subset_38' '97818'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_73,subset_114,subset_111,subset_213,subset_190,subset_206,subset_72,subset_144,subset_68,subset_198,subset_62' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_80,subset_107,subset_105,subset_96,subset_36,subset_97' &
distillation 'subset_78,subset_44,subset_147,subset_131,subset_49,subset_8,subset_29,subset_187,subset_107,subset_177,subset_46' 'subset_56,subset_3,subset_89,subset_131,subset_1,subset_137' '17892'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_33,subset_81,subset_136,subset_138,subset_182,subset_127,subset_133,subset_215,subset_26,subset_39,subset_7' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_47,subset_115,subset_92,subset_69,subset_114,subset_28' &
distillation 'subset_73,subset_114,subset_111,subset_213,subset_190,subset_206,subset_72,subset_144,subset_68,subset_198,subset_62' 'subset_80,subset_107,subset_105,subset_96,subset_36,subset_97' '9150'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_146,subset_121,subset_69,subset_16,subset_157,subset_199,subset_48,subset_27,subset_153,subset_176,subset_132' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_15,subset_122,subset_81,subset_51,subset_24,subset_124' &
distillation 'subset_33,subset_81,subset_136,subset_138,subset_182,subset_127,subset_133,subset_215,subset_26,subset_39,subset_7' 'subset_47,subset_115,subset_92,subset_69,subset_114,subset_28' '36208'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_0,subset_5,subset_112,subset_161,subset_116,subset_123,subset_100,subset_24,subset_66,subset_42,subset_160' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_31,subset_90,subset_41,subset_133,subset_37,subset_74' &
distillation 'subset_146,subset_121,subset_69,subset_16,subset_157,subset_199,subset_48,subset_27,subset_153,subset_176,subset_132' 'subset_15,subset_122,subset_81,subset_51,subset_24,subset_124' '54387'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_220,subset_74,subset_17,subset_79,subset_1,subset_36,subset_96,subset_175,subset_87,subset_179,subset_222' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_83,subset_35,subset_39,subset_75,subset_78,subset_63' &
distillation 'subset_0,subset_5,subset_112,subset_161,subset_116,subset_123,subset_100,subset_24,subset_66,subset_42,subset_160' 'subset_31,subset_90,subset_41,subset_133,subset_37,subset_74' '44548'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_95,subset_60,subset_156,subset_23,subset_104,subset_82,subset_10,subset_88,subset_183,subset_13,subset_28' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_40,subset_113,subset_54,subset_19,subset_116,subset_66' &
distillation 'subset_220,subset_74,subset_17,subset_79,subset_1,subset_36,subset_96,subset_175,subset_87,subset_179,subset_222' 'subset_83,subset_35,subset_39,subset_75,subset_78,subset_63' '66550'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_34,subset_86,subset_159,subset_25,subset_189,subset_204,subset_93,subset_174,subset_4,subset_109,subset_125' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_108,subset_29,subset_42,subset_95,subset_25,subset_129' &
distillation 'subset_95,subset_60,subset_156,subset_23,subset_104,subset_82,subset_10,subset_88,subset_183,subset_13,subset_28' 'subset_40,subset_113,subset_54,subset_19,subset_116,subset_66' '35018'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_188,subset_94,subset_113,subset_31,subset_103,subset_19,subset_168,subset_141,subset_70,subset_221,subset_119' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_59,subset_32,subset_101,subset_11,subset_13,subset_88' &
distillation 'subset_34,subset_86,subset_159,subset_25,subset_189,subset_204,subset_93,subset_174,subset_4,subset_109,subset_125' 'subset_108,subset_29,subset_42,subset_95,subset_25,subset_129' '336'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_54,subset_208,subset_129,subset_56,subset_61,subset_85,subset_124,subset_134,subset_122,subset_21,subset_89' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_52,subset_112,subset_68,subset_49,subset_91,subset_7' &
distillation 'subset_188,subset_94,subset_113,subset_31,subset_103,subset_19,subset_168,subset_141,subset_70,subset_221,subset_119' 'subset_59,subset_32,subset_101,subset_11,subset_13,subset_88' '37072'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_57,subset_91,subset_164,subset_139,subset_130,subset_216,subset_2,subset_137,subset_166,subset_117,subset_126' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_22,subset_126,subset_9,subset_72,subset_33,subset_135' &
distillation 'subset_54,subset_208,subset_129,subset_56,subset_61,subset_85,subset_124,subset_134,subset_122,subset_21,subset_89' 'subset_52,subset_112,subset_68,subset_49,subset_91,subset_7' '95164'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_98,subset_11,subset_18,subset_172,subset_193,subset_43,subset_15,subset_128,subset_151,subset_196,subset_106' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_55,subset_61,subset_53,subset_46,subset_27,subset_119' &
distillation 'subset_57,subset_91,subset_164,subset_139,subset_130,subset_216,subset_2,subset_137,subset_166,subset_117,subset_126' 'subset_22,subset_126,subset_9,subset_72,subset_33,subset_135' '39132'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_192,subset_200,subset_154,subset_35,subset_214,subset_77,subset_205,subset_90,subset_173,subset_163,subset_41' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_76,subset_84,subset_110,subset_14,subset_4,subset_64' &
distillation 'subset_98,subset_11,subset_18,subset_172,subset_193,subset_43,subset_15,subset_128,subset_151,subset_196,subset_106' 'subset_55,subset_61,subset_53,subset_46,subset_27,subset_119' '76931'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_30,subset_158,subset_202,subset_155,subset_50,subset_52,subset_71,subset_83,subset_59,subset_142,subset_84' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_6,subset_136,subset_58,subset_132,subset_98,subset_100' &
distillation 'subset_192,subset_200,subset_154,subset_35,subset_214,subset_77,subset_205,subset_90,subset_173,subset_163,subset_41' 'subset_76,subset_84,subset_110,subset_14,subset_4,subset_64' '76011'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_32,subset_76,subset_97,subset_219,subset_37,subset_92,subset_184,subset_6,subset_162,subset_210,subset_102' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_67,subset_5,subset_0,subset_57,subset_45,subset_130' &
distillation 'subset_30,subset_158,subset_202,subset_155,subset_50,subset_52,subset_71,subset_83,subset_59,subset_142,subset_84' 'subset_6,subset_136,subset_58,subset_132,subset_98,subset_100' '86499'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_118,subset_64,subset_191,subset_135,subset_75,subset_55,subset_140,subset_148,subset_209,subset_181,subset_105' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_73,subset_71,subset_94,subset_128,subset_125,subset_77' &
distillation 'subset_32,subset_76,subset_97,subset_219,subset_37,subset_92,subset_184,subset_6,subset_162,subset_210,subset_102' 'subset_67,subset_5,subset_0,subset_57,subset_45,subset_130' '64178'
distillation 'subset_118,subset_64,subset_191,subset_135,subset_75,subset_55,subset_140,subset_148,subset_209,subset_181,subset_105' 'subset_73,subset_71,subset_94,subset_128,subset_125,subset_77' '64178'
# Epoch 4
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_143,subset_69,subset_16,subset_80,subset_107,subset_124,subset_139,subset_102,subset_125,subset_64,subset_205' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_68,subset_47,subset_132,subset_30,subset_66,subset_6' &
distillation 'subset_118,subset_64,subset_191,subset_135,subset_75,subset_55,subset_140,subset_148,subset_209,subset_181,subset_105' 'subset_73,subset_71,subset_94,subset_128,subset_125,subset_77' '52262'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_106,subset_186,subset_132,subset_73,subset_26,subset_136,subset_187,subset_134,subset_58,subset_177,subset_179' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_71,subset_92,subset_87,subset_134,subset_110,subset_20' &
distillation 'subset_143,subset_69,subset_16,subset_80,subset_107,subset_124,subset_139,subset_102,subset_125,subset_64,subset_205' 'subset_68,subset_47,subset_132,subset_30,subset_66,subset_6' '54524'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_84,subset_88,subset_137,subset_183,subset_55,subset_111,subset_140,subset_45,subset_33,subset_43,subset_222' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_50,subset_37,subset_136,subset_120,subset_129,subset_73' &
distillation 'subset_106,subset_186,subset_132,subset_73,subset_26,subset_136,subset_187,subset_134,subset_58,subset_177,subset_179' 'subset_71,subset_92,subset_87,subset_134,subset_110,subset_20' '95631'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_85,subset_220,subset_212,subset_150,subset_182,subset_83,subset_92,subset_74,subset_208,subset_89,subset_151' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_93,subset_85,subset_52,subset_10,subset_112,subset_54' &
distillation 'subset_84,subset_88,subset_137,subset_183,subset_55,subset_111,subset_140,subset_45,subset_33,subset_43,subset_222' 'subset_50,subset_37,subset_136,subset_120,subset_129,subset_73' '12463'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_121,subset_61,subset_99,subset_9,subset_135,subset_62,subset_110,subset_129,subset_116,subset_51,subset_57' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_111,subset_114,subset_122,subset_29,subset_21,subset_81' &
distillation 'subset_85,subset_220,subset_212,subset_150,subset_182,subset_83,subset_92,subset_74,subset_208,subset_89,subset_151' 'subset_93,subset_85,subset_52,subset_10,subset_112,subset_54' '40966'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_178,subset_14,subset_115,subset_127,subset_198,subset_20,subset_131,subset_173,subset_112,subset_162,subset_218' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_75,subset_125,subset_31,subset_46,subset_74,subset_126' &
distillation 'subset_121,subset_61,subset_99,subset_9,subset_135,subset_62,subset_110,subset_129,subset_116,subset_51,subset_57' 'subset_111,subset_114,subset_122,subset_29,subset_21,subset_81' '55936'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_105,subset_167,subset_70,subset_192,subset_165,subset_195,subset_15,subset_1,subset_22,subset_120,subset_123' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_123,subset_35,subset_61,subset_12,subset_64,subset_43' &
distillation 'subset_178,subset_14,subset_115,subset_127,subset_198,subset_20,subset_131,subset_173,subset_112,subset_162,subset_218' 'subset_75,subset_125,subset_31,subset_46,subset_74,subset_126' '40963'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_175,subset_217,subset_77,subset_68,subset_146,subset_38,subset_133,subset_196,subset_207,subset_29,subset_200' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_63,subset_7,subset_108,subset_16,subset_32,subset_133' &
distillation 'subset_105,subset_167,subset_70,subset_192,subset_165,subset_195,subset_15,subset_1,subset_22,subset_120,subset_123' 'subset_123,subset_35,subset_61,subset_12,subset_64,subset_43' '87177'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_180,subset_119,subset_5,subset_75,subset_100,subset_44,subset_164,subset_118,subset_53,subset_171,subset_109' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_83,subset_19,subset_115,subset_9,subset_38,subset_90' &
distillation 'subset_175,subset_217,subset_77,subset_68,subset_146,subset_38,subset_133,subset_196,subset_207,subset_29,subset_200' 'subset_63,subset_7,subset_108,subset_16,subset_32,subset_133' '33408'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_8,subset_30,subset_87,subset_181,subset_86,subset_209,subset_46,subset_189,subset_52,subset_54,subset_94' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_1,subset_42,subset_65,subset_113,subset_67,subset_116' &
distillation 'subset_180,subset_119,subset_5,subset_75,subset_100,subset_44,subset_164,subset_118,subset_53,subset_171,subset_109' 'subset_83,subset_19,subset_115,subset_9,subset_38,subset_90' '49060'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_114,subset_117,subset_188,subset_138,subset_128,subset_158,subset_13,subset_50,subset_79,subset_82,subset_108' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_130,subset_45,subset_135,subset_119,subset_96,subset_101' &
distillation 'subset_8,subset_30,subset_87,subset_181,subset_86,subset_209,subset_46,subset_189,subset_52,subset_54,subset_94' 'subset_1,subset_42,subset_65,subset_113,subset_67,subset_116' '20006'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_148,subset_166,subset_91,subset_202,subset_126,subset_4,subset_184,subset_35,subset_93,subset_210,subset_98' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_76,subset_28,subset_128,subset_106,subset_2,subset_88' &
distillation 'subset_114,subset_117,subset_188,subset_138,subset_128,subset_158,subset_13,subset_50,subset_79,subset_82,subset_108' 'subset_130,subset_45,subset_135,subset_119,subset_96,subset_101' '90023'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_172,subset_78,subset_185,subset_147,subset_37,subset_31,subset_144,subset_101,subset_95,subset_24,subset_161' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_34,subset_86,subset_70,subset_105,subset_57,subset_53' &
distillation 'subset_148,subset_166,subset_91,subset_202,subset_126,subset_4,subset_184,subset_35,subset_93,subset_210,subset_98' 'subset_76,subset_28,subset_128,subset_106,subset_2,subset_88' '62163'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_39,subset_65,subset_48,subset_219,subset_40,subset_7,subset_203,subset_23,subset_21,subset_90,subset_60' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_48,subset_99,subset_23,subset_84,subset_26,subset_103' &
distillation 'subset_172,subset_78,subset_185,subset_147,subset_37,subset_31,subset_144,subset_101,subset_95,subset_24,subset_161' 'subset_34,subset_86,subset_70,subset_105,subset_57,subset_53' '8799'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_47,subset_213,subset_215,subset_206,subset_3,subset_130,subset_76,subset_168,subset_25,subset_2,subset_97' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_137,subset_4,subset_11,subset_55,subset_25,subset_60' &
distillation 'subset_39,subset_65,subset_48,subset_219,subset_40,subset_7,subset_203,subset_23,subset_21,subset_90,subset_60' 'subset_48,subset_99,subset_23,subset_84,subset_26,subset_103' '11957'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_142,subset_103,subset_41,subset_197,subset_71,subset_17,subset_216,subset_176,subset_81,subset_28,subset_67' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_89,subset_59,subset_121,subset_107,subset_15,subset_127' &
distillation 'subset_47,subset_213,subset_215,subset_206,subset_3,subset_130,subset_76,subset_168,subset_25,subset_2,subset_97' 'subset_137,subset_4,subset_11,subset_55,subset_25,subset_60' '11188'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_170,subset_18,subset_36,subset_152,subset_201,subset_191,subset_113,subset_163,subset_156,subset_63,subset_27' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_118,subset_13,subset_49,subset_0,subset_102,subset_22' &
distillation 'subset_142,subset_103,subset_41,subset_197,subset_71,subset_17,subset_216,subset_176,subset_81,subset_28,subset_67' 'subset_89,subset_59,subset_121,subset_107,subset_15,subset_127' '12219'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_122,subset_174,subset_149,subset_145,subset_204,subset_194,subset_11,subset_193,subset_6,subset_42,subset_214' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_17,subset_78,subset_117,subset_56,subset_44,subset_41' &
distillation 'subset_170,subset_18,subset_36,subset_152,subset_201,subset_191,subset_113,subset_163,subset_156,subset_63,subset_27' 'subset_118,subset_13,subset_49,subset_0,subset_102,subset_22' '56606'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_34,subset_155,subset_157,subset_12,subset_96,subset_32,subset_190,subset_160,subset_56,subset_72,subset_154' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_82,subset_51,subset_91,subset_95,subset_79,subset_72' &
distillation 'subset_122,subset_174,subset_149,subset_145,subset_204,subset_194,subset_11,subset_193,subset_6,subset_42,subset_214' 'subset_17,subset_78,subset_117,subset_56,subset_44,subset_41' '12656'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_49,subset_199,subset_10,subset_66,subset_141,subset_59,subset_221,subset_153,subset_0,subset_159,subset_19' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_24,subset_5,subset_80,subset_77,subset_39' &
distillation 'subset_34,subset_155,subset_157,subset_12,subset_96,subset_32,subset_190,subset_160,subset_56,subset_72,subset_154' 'subset_82,subset_51,subset_91,subset_95,subset_79,subset_72' '97594'
distillation 'subset_49,subset_199,subset_10,subset_66,subset_141,subset_59,subset_221,subset_153,subset_0,subset_159,subset_19' 'subset_109,subset_24,subset_5,subset_80,subset_77,subset_39' '97594'
# Epoch 5
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_108,subset_105,subset_103,subset_80,subset_175,subset_125,subset_76,subset_131,subset_86,subset_220,subset_185' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_79,subset_68,subset_30,subset_51,subset_91,subset_22' &
distillation 'subset_49,subset_199,subset_10,subset_66,subset_141,subset_59,subset_221,subset_153,subset_0,subset_159,subset_19' 'subset_109,subset_24,subset_5,subset_80,subset_77,subset_39' '84040'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_163,subset_3,subset_96,subset_144,subset_203,subset_20,subset_160,subset_117,subset_116,subset_165,subset_151' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_47,subset_116,subset_18,subset_117,subset_9,subset_42' &
distillation 'subset_108,subset_105,subset_103,subset_80,subset_175,subset_125,subset_76,subset_131,subset_86,subset_220,subset_185' 'subset_79,subset_68,subset_30,subset_51,subset_91,subset_22' '62991'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_24,subset_37,subset_68,subset_33,subset_79,subset_153,subset_134,subset_137,subset_198,subset_218,subset_212' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_106,subset_14,subset_126,subset_103,subset_131,subset_120' &
distillation 'subset_163,subset_3,subset_96,subset_144,subset_203,subset_20,subset_160,subset_117,subset_116,subset_165,subset_151' 'subset_47,subset_116,subset_18,subset_117,subset_9,subset_42' '48655'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_25,subset_188,subset_67,subset_87,subset_214,subset_89,subset_170,subset_209,subset_7,subset_184,subset_121' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_114,subset_75,subset_73,subset_0,subset_29,subset_81' &
distillation 'subset_24,subset_37,subset_68,subset_33,subset_79,subset_153,subset_134,subset_137,subset_198,subset_218,subset_212' 'subset_106,subset_14,subset_126,subset_103,subset_131,subset_120' '72684'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_8,subset_18,subset_210,subset_11,subset_158,subset_38,subset_47,subset_107,subset_205,subset_62,subset_14' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_105,subset_64,subset_45,subset_101,subset_124,subset_56' &
distillation 'subset_25,subset_188,subset_67,subset_87,subset_214,subset_89,subset_170,subset_209,subset_7,subset_184,subset_121' 'subset_114,subset_75,subset_73,subset_0,subset_29,subset_81' '13446'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_31,subset_100,subset_9,subset_58,subset_44,subset_183,subset_90,subset_19,subset_22,subset_166,subset_10' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_127,subset_135,subset_122,subset_16,subset_21,subset_35' &
distillation 'subset_8,subset_18,subset_210,subset_11,subset_158,subset_38,subset_47,subset_107,subset_205,subset_62,subset_14' 'subset_105,subset_64,subset_45,subset_101,subset_124,subset_56' '93244'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_34,subset_126,subset_106,subset_142,subset_141,subset_63,subset_199,subset_75,subset_217,subset_73,subset_201' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_97,subset_4,subset_118,subset_111,subset_61,subset_90' &
distillation 'subset_31,subset_100,subset_9,subset_58,subset_44,subset_183,subset_90,subset_19,subset_22,subset_166,subset_10' 'subset_127,subset_135,subset_122,subset_16,subset_21,subset_35' '67621'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_51,subset_129,subset_146,subset_42,subset_32,subset_113,subset_41,subset_127,subset_16,subset_30,subset_82' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_2,subset_67,subset_128,subset_119,subset_46,subset_74' &
distillation 'subset_34,subset_126,subset_106,subset_142,subset_141,subset_63,subset_199,subset_75,subset_217,subset_73,subset_201' 'subset_97,subset_4,subset_118,subset_111,subset_61,subset_90' '16341'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_50,subset_64,subset_61,subset_148,subset_27,subset_162,subset_172,subset_157,subset_81,subset_135,subset_136' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_5,subset_121,subset_107,subset_66,subset_134,subset_25' &
distillation 'subset_51,subset_129,subset_146,subset_42,subset_32,subset_113,subset_41,subset_127,subset_16,subset_30,subset_82' 'subset_2,subset_67,subset_128,subset_119,subset_46,subset_74' '37363'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_84,subset_78,subset_169,subset_112,subset_128,subset_0,subset_208,subset_182,subset_222,subset_171,subset_168' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_40,subset_36,subset_82,subset_137,subset_10,subset_39' &
distillation 'subset_50,subset_64,subset_61,subset_148,subset_27,subset_162,subset_172,subset_157,subset_81,subset_135,subset_136' 'subset_5,subset_121,subset_107,subset_66,subset_134,subset_25' '10993'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_109,subset_204,subset_83,subset_155,subset_143,subset_191,subset_123,subset_110,subset_35,subset_124,subset_2' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_98,subset_109,subset_27,subset_132,subset_102,subset_1' &
distillation 'subset_84,subset_78,subset_169,subset_112,subset_128,subset_0,subset_208,subset_182,subset_222,subset_171,subset_168' 'subset_40,subset_36,subset_82,subset_137,subset_10,subset_39' '21012'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_207,subset_167,subset_173,subset_120,subset_4,subset_28,subset_15,subset_140,subset_114,subset_180,subset_122' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_20,subset_34,subset_99,subset_58,subset_84,subset_17' &
distillation 'subset_109,subset_204,subset_83,subset_155,subset_143,subset_191,subset_123,subset_110,subset_35,subset_124,subset_2' 'subset_98,subset_109,subset_27,subset_132,subset_102,subset_1' '35755'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_98,subset_215,subset_95,subset_189,subset_101,subset_202,subset_56,subset_138,subset_154,subset_177,subset_186' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_115,subset_12,subset_108,subset_93,subset_70' &
distillation 'subset_207,subset_167,subset_173,subset_120,subset_4,subset_28,subset_15,subset_140,subset_114,subset_180,subset_122' 'subset_20,subset_34,subset_99,subset_58,subset_84,subset_17' '58893'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_21,subset_6,subset_190,subset_179,subset_48,subset_133,subset_71,subset_174,subset_74,subset_200,subset_13' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_55,subset_33,subset_62,subset_31,subset_69' &
distillation 'subset_98,subset_215,subset_95,subset_189,subset_101,subset_202,subset_56,subset_138,subset_154,subset_177,subset_186' 'subset_129,subset_115,subset_12,subset_108,subset_93,subset_70' '67286'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_52,subset_53,subset_178,subset_147,subset_88,subset_152,subset_36,subset_187,subset_181,subset_65,subset_150' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_13,subset_88,subset_50,subset_60,subset_87,subset_63' &
distillation 'subset_21,subset_6,subset_190,subset_179,subset_48,subset_133,subset_71,subset_174,subset_74,subset_200,subset_13' 'subset_72,subset_55,subset_33,subset_62,subset_31,subset_69' '19313'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_66,subset_69,subset_145,subset_29,subset_12,subset_192,subset_99,subset_102,subset_70,subset_193,subset_213' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_95,subset_23,subset_71,subset_57,subset_11,subset_80' &
distillation 'subset_52,subset_53,subset_178,subset_147,subset_88,subset_152,subset_36,subset_187,subset_181,subset_65,subset_150' 'subset_13,subset_88,subset_50,subset_60,subset_87,subset_63' '57340'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_139,subset_211,subset_97,subset_176,subset_206,subset_130,subset_59,subset_94,subset_104,subset_219,subset_195' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_38,subset_19,subset_123,subset_133,subset_32,subset_52' &
distillation 'subset_66,subset_69,subset_145,subset_29,subset_12,subset_192,subset_99,subset_102,subset_70,subset_193,subset_213' 'subset_95,subset_23,subset_71,subset_57,subset_11,subset_80' '12021'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_45,subset_164,subset_159,subset_119,subset_111,subset_115,subset_91,subset_197,subset_92,subset_57,subset_93' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_48,subset_76,subset_28,subset_44,subset_65,subset_89' &
distillation 'subset_139,subset_211,subset_97,subset_176,subset_206,subset_130,subset_59,subset_94,subset_104,subset_219,subset_195' 'subset_38,subset_19,subset_123,subset_133,subset_32,subset_52' '29102'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_40,subset_55,subset_49,subset_77,subset_60,subset_1,subset_132,subset_156,subset_54,subset_194,subset_17' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_125,subset_6,subset_112,subset_26,subset_8,subset_100' &
distillation 'subset_45,subset_164,subset_159,subset_119,subset_111,subset_115,subset_91,subset_197,subset_92,subset_57,subset_93' 'subset_48,subset_76,subset_28,subset_44,subset_65,subset_89' '59130'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_5,subset_46,subset_43,subset_216,subset_196,subset_221,subset_39,subset_23,subset_26,subset_161,subset_85' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_59,subset_7,subset_41,subset_104,subset_78,subset_37' &
distillation 'subset_40,subset_55,subset_49,subset_77,subset_60,subset_1,subset_132,subset_156,subset_54,subset_194,subset_17' 'subset_125,subset_6,subset_112,subset_26,subset_8,subset_100' '45820'
distillation 'subset_5,subset_46,subset_43,subset_216,subset_196,subset_221,subset_39,subset_23,subset_26,subset_161,subset_85' 'subset_59,subset_7,subset_41,subset_104,subset_78,subset_37' '45820'
# Epoch 6
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_99,subset_25,subset_36,subset_137,subset_130,subset_123,subset_88,subset_101,subset_205,subset_176,subset_35' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_87,subset_13,subset_133,subset_105,subset_67,subset_97' &
distillation 'subset_5,subset_46,subset_43,subset_216,subset_196,subset_221,subset_39,subset_23,subset_26,subset_161,subset_85' 'subset_59,subset_7,subset_41,subset_104,subset_78,subset_37' '90257'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_19,subset_55,subset_30,subset_63,subset_198,subset_74,subset_146,subset_126,subset_144,subset_172,subset_213' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_6,subset_26,subset_50,subset_69,subset_2,subset_91' &
distillation 'subset_99,subset_25,subset_36,subset_137,subset_130,subset_123,subset_88,subset_101,subset_205,subset_176,subset_35' 'subset_87,subset_13,subset_133,subset_105,subset_67,subset_97' '7852'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_40,subset_102,subset_211,subset_180,subset_12,subset_191,subset_178,subset_43,subset_135,subset_109,subset_158' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_5,subset_90,subset_121,subset_19,subset_38,subset_14' &
distillation 'subset_19,subset_55,subset_30,subset_63,subset_198,subset_74,subset_146,subset_126,subset_144,subset_172,subset_213' 'subset_6,subset_26,subset_50,subset_69,subset_2,subset_91' '20719'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_129,subset_156,subset_115,subset_62,subset_162,subset_199,subset_112,subset_71,subset_80,subset_94,subset_204' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_110,subset_47,subset_113,subset_123,subset_77,subset_101' &
distillation 'subset_40,subset_102,subset_211,subset_180,subset_12,subset_191,subset_178,subset_43,subset_135,subset_109,subset_158' 'subset_5,subset_90,subset_121,subset_19,subset_38,subset_14' '57619'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_145,subset_39,subset_81,subset_210,subset_103,subset_93,subset_207,subset_216,subset_8,subset_106,subset_153' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_30,subset_79,subset_11,subset_68,subset_65,subset_56' &
distillation 'subset_129,subset_156,subset_115,subset_62,subset_162,subset_199,subset_112,subset_71,subset_80,subset_94,subset_204' 'subset_110,subset_47,subset_113,subset_123,subset_77,subset_101' '54476'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_98,subset_45,subset_192,subset_116,subset_117,subset_6,subset_111,subset_78,subset_72,subset_7,subset_208' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_45,subset_0,subset_112,subset_98,subset_93,subset_27' &
distillation 'subset_145,subset_39,subset_81,subset_210,subset_103,subset_93,subset_207,subset_216,subset_8,subset_106,subset_153' 'subset_30,subset_79,subset_11,subset_68,subset_65,subset_56' '63471'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_201,subset_46,subset_141,subset_14,subset_185,subset_127,subset_34,subset_70,subset_73,subset_22,subset_155' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_33,subset_24,subset_54,subset_31,subset_80,subset_35' &
distillation 'subset_98,subset_45,subset_192,subset_116,subset_117,subset_6,subset_111,subset_78,subset_72,subset_7,subset_208' 'subset_45,subset_0,subset_112,subset_98,subset_93,subset_27' '60875'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_120,subset_38,subset_100,subset_107,subset_164,subset_29,subset_203,subset_161,subset_184,subset_41,subset_149' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_106,subset_102,subset_22,subset_83,subset_94' &
distillation 'subset_201,subset_46,subset_141,subset_14,subset_185,subset_127,subset_34,subset_70,subset_73,subset_22,subset_155' 'subset_33,subset_24,subset_54,subset_31,subset_80,subset_35' '26741'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_28,subset_66,subset_68,subset_167,subset_195,subset_212,subset_57,subset_4,subset_47,subset_159,subset_170' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_48,subset_126,subset_119,subset_89,subset_55,subset_9' &
distillation 'subset_120,subset_38,subset_100,subset_107,subset_164,subset_29,subset_203,subset_161,subset_184,subset_41,subset_149' 'subset_129,subset_106,subset_102,subset_22,subset_83,subset_94' '44589'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_209,subset_83,subset_173,subset_85,subset_186,subset_5,subset_132,subset_104,subset_58,subset_124,subset_193' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_75,subset_17,subset_16,subset_116,subset_25,subset_71' &
distillation 'subset_28,subset_66,subset_68,subset_167,subset_195,subset_212,subset_57,subset_4,subset_47,subset_159,subset_170' 'subset_48,subset_126,subset_119,subset_89,subset_55,subset_9' '79516'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_86,subset_215,subset_128,subset_37,subset_42,subset_150,subset_64,subset_44,subset_82,subset_56,subset_13' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_57,subset_10,subset_122,subset_36,subset_99,subset_81' &
distillation 'subset_209,subset_83,subset_173,subset_85,subset_186,subset_5,subset_132,subset_104,subset_58,subset_124,subset_193' 'subset_75,subset_17,subset_16,subset_116,subset_25,subset_71' '18830'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_24,subset_113,subset_0,subset_121,subset_142,subset_18,subset_52,subset_188,subset_54,subset_79,subset_219' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_108,subset_130,subset_86,subset_37,subset_43,subset_12' &
distillation 'subset_86,subset_215,subset_128,subset_37,subset_42,subset_150,subset_64,subset_44,subset_82,subset_56,subset_13' 'subset_57,subset_10,subset_122,subset_36,subset_99,subset_81' '40970'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_50,subset_133,subset_148,subset_194,subset_166,subset_196,subset_21,subset_163,subset_92,subset_84,subset_125' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_104,subset_41,subset_117,subset_28,subset_70,subset_134' &
distillation 'subset_24,subset_113,subset_0,subset_121,subset_142,subset_18,subset_52,subset_188,subset_54,subset_79,subset_219' 'subset_108,subset_130,subset_86,subset_37,subset_43,subset_12' '94157'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_177,subset_190,subset_114,subset_138,subset_174,subset_160,subset_27,subset_95,subset_151,subset_11,subset_118' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_15,subset_109,subset_107,subset_111,subset_100,subset_92' &
distillation 'subset_50,subset_133,subset_148,subset_194,subset_166,subset_196,subset_21,subset_163,subset_92,subset_84,subset_125' 'subset_104,subset_41,subset_117,subset_28,subset_70,subset_134' '41853'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_179,subset_181,subset_61,subset_91,subset_108,subset_168,subset_217,subset_87,subset_139,subset_48,subset_122' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_131,subset_44,subset_39,subset_74,subset_3' &
distillation 'subset_177,subset_190,subset_114,subset_138,subset_174,subset_160,subset_27,subset_95,subset_151,subset_11,subset_118' 'subset_15,subset_109,subset_107,subset_111,subset_100,subset_92' '96233'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_187,subset_119,subset_202,subset_200,subset_15,subset_17,subset_20,subset_110,subset_152,subset_65,subset_222' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_8,subset_32,subset_51,subset_132,subset_136,subset_120' &
distillation 'subset_179,subset_181,subset_61,subset_91,subset_108,subset_168,subset_217,subset_87,subset_139,subset_48,subset_122' 'subset_72,subset_131,subset_44,subset_39,subset_74,subset_3' '45268'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_90,subset_3,subset_218,subset_51,subset_23,subset_77,subset_134,subset_171,subset_189,subset_53,subset_154' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_64,subset_103,subset_124,subset_1,subset_46,subset_52' &
distillation 'subset_187,subset_119,subset_202,subset_200,subset_15,subset_17,subset_20,subset_110,subset_152,subset_65,subset_222' 'subset_8,subset_32,subset_51,subset_132,subset_136,subset_120' '52260'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_89,subset_31,subset_10,subset_175,subset_221,subset_105,subset_147,subset_143,subset_49,subset_206,subset_197' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_49,subset_62,subset_63,subset_85,subset_96,subset_73' &
distillation 'subset_90,subset_3,subset_218,subset_51,subset_23,subset_77,subset_134,subset_171,subset_189,subset_53,subset_154' 'subset_64,subset_103,subset_124,subset_1,subset_46,subset_52' '17139'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_1,subset_97,subset_2,subset_75,subset_220,subset_67,subset_214,subset_60,subset_16,subset_69,subset_59' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_78,subset_118,subset_4,subset_58,subset_84,subset_18' &
distillation 'subset_89,subset_31,subset_10,subset_175,subset_221,subset_105,subset_147,subset_143,subset_49,subset_206,subset_197' 'subset_49,subset_62,subset_63,subset_85,subset_96,subset_73' '99701'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_33,subset_32,subset_131,subset_9,subset_26,subset_165,subset_136,subset_183,subset_157,subset_140,subset_169' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_95,subset_128,subset_88,subset_61,subset_125,subset_60' &
distillation 'subset_1,subset_97,subset_2,subset_75,subset_220,subset_67,subset_214,subset_60,subset_16,subset_69,subset_59' 'subset_78,subset_118,subset_4,subset_58,subset_84,subset_18' '48567'
distillation 'subset_33,subset_32,subset_131,subset_9,subset_26,subset_165,subset_136,subset_183,subset_157,subset_140,subset_169' 'subset_95,subset_128,subset_88,subset_61,subset_125,subset_60' '48567'
# Epoch 7
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_101,subset_151,subset_60,subset_108,subset_192,subset_220,subset_10,subset_25,subset_180,subset_99,subset_3' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_28,subset_75,subset_111,subset_109,subset_47,subset_83' &
distillation 'subset_33,subset_32,subset_131,subset_9,subset_26,subset_165,subset_136,subset_183,subset_157,subset_140,subset_169' 'subset_95,subset_128,subset_88,subset_61,subset_125,subset_60' '59634'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_171,subset_65,subset_175,subset_66,subset_14,subset_162,subset_17,subset_47,subset_116,subset_133,subset_185' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_12,subset_102,subset_95,subset_62,subset_72,subset_8' &
distillation 'subset_101,subset_151,subset_60,subset_108,subset_192,subset_220,subset_10,subset_25,subset_180,subset_99,subset_3' 'subset_28,subset_75,subset_111,subset_109,subset_47,subset_83' '98007'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_215,subset_117,subset_128,subset_194,subset_95,subset_178,subset_82,subset_1,subset_218,subset_120,subset_216' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_0,subset_85,subset_133,subset_88,subset_117,subset_105' &
distillation 'subset_171,subset_65,subset_175,subset_66,subset_14,subset_162,subset_17,subset_47,subset_116,subset_133,subset_185' 'subset_12,subset_102,subset_95,subset_62,subset_72,subset_8' '79395'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_161,subset_169,subset_146,subset_198,subset_23,subset_21,subset_159,subset_74,subset_40,subset_145,subset_29' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_1,subset_49,subset_126,subset_131,subset_103,subset_59' &
distillation 'subset_215,subset_117,subset_128,subset_194,subset_95,subset_178,subset_82,subset_1,subset_218,subset_120,subset_216' 'subset_0,subset_85,subset_133,subset_88,subset_117,subset_105' '48971'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_200,subset_173,subset_148,subset_42,subset_124,subset_15,subset_160,subset_222,subset_113,subset_221,subset_87' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_11,subset_44,subset_16,subset_69,subset_2,subset_135' &
distillation 'subset_161,subset_169,subset_146,subset_198,subset_23,subset_21,subset_159,subset_74,subset_40,subset_145,subset_29' 'subset_1,subset_49,subset_126,subset_131,subset_103,subset_59' '54505'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_193,subset_205,subset_39,subset_61,subset_81,subset_187,subset_153,subset_75,subset_167,subset_201,subset_12' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_98,subset_65,subset_17,subset_73,subset_38,subset_58' &
distillation 'subset_200,subset_173,subset_148,subset_42,subset_124,subset_15,subset_160,subset_222,subset_113,subset_221,subset_87' 'subset_11,subset_44,subset_16,subset_69,subset_2,subset_135' '92004'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_8,subset_144,subset_209,subset_0,subset_45,subset_132,subset_138,subset_217,subset_149,subset_130,subset_183' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_19,subset_53,subset_45,subset_50,subset_89,subset_6' &
distillation 'subset_193,subset_205,subset_39,subset_61,subset_81,subset_187,subset_153,subset_75,subset_167,subset_201,subset_12' 'subset_98,subset_65,subset_17,subset_73,subset_38,subset_58' '71989'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_179,subset_97,subset_100,subset_142,subset_90,subset_213,subset_53,subset_190,subset_88,subset_43,subset_181' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_108,subset_23,subset_97,subset_81,subset_52,subset_40' &
distillation 'subset_8,subset_144,subset_209,subset_0,subset_45,subset_132,subset_138,subset_217,subset_149,subset_130,subset_183' 'subset_19,subset_53,subset_45,subset_50,subset_89,subset_6' '61712'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_118,subset_67,subset_129,subset_37,subset_155,subset_137,subset_2,subset_28,subset_69,subset_112,subset_33' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_78,subset_80,subset_31,subset_4,subset_42,subset_48' &
distillation 'subset_179,subset_97,subset_100,subset_142,subset_90,subset_213,subset_53,subset_190,subset_88,subset_43,subset_181' 'subset_108,subset_23,subset_97,subset_81,subset_52,subset_40' '99267'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_106,subset_63,subset_168,subset_94,subset_210,subset_20,subset_5,subset_6,subset_154,subset_51,subset_134' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_106,subset_107,subset_110,subset_74,subset_123,subset_94' &
distillation 'subset_118,subset_67,subset_129,subset_37,subset_155,subset_137,subset_2,subset_28,subset_69,subset_112,subset_33' 'subset_78,subset_80,subset_31,subset_4,subset_42,subset_48' '70453'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_78,subset_136,subset_208,subset_104,subset_7,subset_204,subset_22,subset_166,subset_16,subset_57,subset_121' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_61,subset_46,subset_22,subset_112,subset_70,subset_128' &
distillation 'subset_106,subset_63,subset_168,subset_94,subset_210,subset_20,subset_5,subset_6,subset_154,subset_51,subset_134' 'subset_106,subset_107,subset_110,subset_74,subset_123,subset_94' '87044'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_163,subset_172,subset_197,subset_105,subset_143,subset_80,subset_50,subset_199,subset_64,subset_24,subset_184' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_15,subset_121,subset_30,subset_93,subset_21,subset_99' &
distillation 'subset_78,subset_136,subset_208,subset_104,subset_7,subset_204,subset_22,subset_166,subset_16,subset_57,subset_121' 'subset_61,subset_46,subset_22,subset_112,subset_70,subset_128' '28627'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_115,subset_176,subset_48,subset_35,subset_188,subset_139,subset_79,subset_207,subset_9,subset_122,subset_125' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_125,subset_76,subset_130,subset_20,subset_25,subset_63' &
distillation 'subset_163,subset_172,subset_197,subset_105,subset_143,subset_80,subset_50,subset_199,subset_64,subset_24,subset_184' 'subset_15,subset_121,subset_30,subset_93,subset_21,subset_99' '99858'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_41,subset_4,subset_31,subset_131,subset_158,subset_135,subset_127,subset_92,subset_165,subset_89,subset_59' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_7,subset_114,subset_134,subset_104,subset_54,subset_13' &
distillation 'subset_115,subset_176,subset_48,subset_35,subset_188,subset_139,subset_79,subset_207,subset_9,subset_122,subset_125' 'subset_125,subset_76,subset_130,subset_20,subset_25,subset_63' '32435'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_126,subset_152,subset_84,subset_195,subset_93,subset_182,subset_114,subset_30,subset_85,subset_19,subset_119' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_24,subset_101,subset_36,subset_119,subset_132' &
distillation 'subset_41,subset_4,subset_31,subset_131,subset_158,subset_135,subset_127,subset_92,subset_165,subset_89,subset_59' 'subset_7,subset_114,subset_134,subset_104,subset_54,subset_13' '89173'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_186,subset_212,subset_46,subset_156,subset_86,subset_96,subset_36,subset_196,subset_70,subset_13,subset_110' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_120,subset_9,subset_77,subset_67,subset_51,subset_136' &
distillation 'subset_126,subset_152,subset_84,subset_195,subset_93,subset_182,subset_114,subset_30,subset_85,subset_19,subset_119' 'subset_3,subset_24,subset_101,subset_36,subset_119,subset_132' '98925'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_202,subset_11,subset_18,subset_34,subset_77,subset_32,subset_68,subset_27,subset_98,subset_140,subset_44' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_86,subset_56,subset_41,subset_118,subset_91,subset_60' &
distillation 'subset_186,subset_212,subset_46,subset_156,subset_86,subset_96,subset_36,subset_196,subset_70,subset_13,subset_110' 'subset_120,subset_9,subset_77,subset_67,subset_51,subset_136' '78094'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_54,subset_83,subset_102,subset_123,subset_147,subset_38,subset_206,subset_157,subset_76,subset_52,subset_214' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_26,subset_100,subset_96,subset_35,subset_33,subset_64' &
distillation 'subset_202,subset_11,subset_18,subset_34,subset_77,subset_32,subset_68,subset_27,subset_98,subset_140,subset_44' 'subset_86,subset_56,subset_41,subset_118,subset_91,subset_60' '10730'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_49,subset_103,subset_107,subset_189,subset_174,subset_203,subset_71,subset_26,subset_58,subset_91,subset_170' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_79,subset_18,subset_115,subset_82,subset_84,subset_92' &
distillation 'subset_54,subset_83,subset_102,subset_123,subset_147,subset_38,subset_206,subset_157,subset_76,subset_52,subset_214' 'subset_26,subset_100,subset_96,subset_35,subset_33,subset_64' '68879'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_73,subset_55,subset_211,subset_62,subset_72,subset_150,subset_141,subset_109,subset_56,subset_111,subset_164' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_29,subset_113,subset_43,subset_87,subset_14,subset_124' &
distillation 'subset_49,subset_103,subset_107,subset_189,subset_174,subset_203,subset_71,subset_26,subset_58,subset_91,subset_170' 'subset_79,subset_18,subset_115,subset_82,subset_84,subset_92' '58556'
distillation 'subset_73,subset_55,subset_211,subset_62,subset_72,subset_150,subset_141,subset_109,subset_56,subset_111,subset_164' 'subset_29,subset_113,subset_43,subset_87,subset_14,subset_124' '58556'
# Epoch 8
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_83,subset_151,subset_194,subset_98,subset_90,subset_202,subset_48,subset_32,subset_204,subset_138,subset_160' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_48,subset_1,subset_25,subset_82,subset_137,subset_118' &
distillation 'subset_73,subset_55,subset_211,subset_62,subset_72,subset_150,subset_141,subset_109,subset_56,subset_111,subset_164' 'subset_29,subset_113,subset_43,subset_87,subset_14,subset_124' '2052'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_42,subset_37,subset_84,subset_198,subset_170,subset_195,subset_18,subset_213,subset_75,subset_54,subset_210' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_6,subset_14,subset_11,subset_132,subset_76,subset_122' &
distillation 'subset_83,subset_151,subset_194,subset_98,subset_90,subset_202,subset_48,subset_32,subset_204,subset_138,subset_160' 'subset_48,subset_1,subset_25,subset_82,subset_137,subset_118' '27152'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_94,subset_58,subset_163,subset_211,subset_81,subset_3,subset_122,subset_72,subset_125,subset_186,subset_209' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_107,subset_45,subset_2,subset_16,subset_52,subset_109' &
distillation 'subset_42,subset_37,subset_84,subset_198,subset_170,subset_195,subset_18,subset_213,subset_75,subset_54,subset_210' 'subset_6,subset_14,subset_11,subset_132,subset_76,subset_122' '76943'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_172,subset_60,subset_15,subset_212,subset_150,subset_105,subset_152,subset_29,subset_126,subset_179,subset_26' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_85,subset_74,subset_130,subset_39,subset_91' &
distillation 'subset_94,subset_58,subset_163,subset_211,subset_81,subset_3,subset_122,subset_72,subset_125,subset_186,subset_209' 'subset_107,subset_45,subset_2,subset_16,subset_52,subset_109' '18899'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_114,subset_185,subset_175,subset_70,subset_36,subset_128,subset_103,subset_206,subset_34,subset_50,subset_80' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_26,subset_126,subset_99,subset_66,subset_94,subset_98' &
distillation 'subset_172,subset_60,subset_15,subset_212,subset_150,subset_105,subset_152,subset_29,subset_126,subset_179,subset_26' 'subset_72,subset_85,subset_74,subset_130,subset_39,subset_91' '98599'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_88,subset_108,subset_68,subset_208,subset_99,subset_89,subset_100,subset_134,subset_107,subset_92,subset_187' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_80,subset_89,subset_38,subset_128,subset_32,subset_114' &
distillation 'subset_114,subset_185,subset_175,subset_70,subset_36,subset_128,subset_103,subset_206,subset_34,subset_50,subset_80' 'subset_26,subset_126,subset_99,subset_66,subset_94,subset_98' '93416'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_49,subset_149,subset_85,subset_164,subset_181,subset_221,subset_33,subset_144,subset_115,subset_200,subset_123' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_133,subset_31,subset_87,subset_127,subset_20,subset_124' &
distillation 'subset_88,subset_108,subset_68,subset_208,subset_99,subset_89,subset_100,subset_134,subset_107,subset_92,subset_187' 'subset_80,subset_89,subset_38,subset_128,subset_32,subset_114' '52205'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_165,subset_91,subset_87,subset_142,subset_215,subset_130,subset_132,subset_5,subset_146,subset_191,subset_174' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_83,subset_129,subset_113,subset_103,subset_117,subset_61' &
distillation 'subset_49,subset_149,subset_85,subset_164,subset_181,subset_221,subset_33,subset_144,subset_115,subset_200,subset_123' 'subset_133,subset_31,subset_87,subset_127,subset_20,subset_124' '10084'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_38,subset_8,subset_190,subset_11,subset_73,subset_59,subset_40,subset_57,subset_86,subset_147,subset_74' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_50,subset_0,subset_88,subset_119,subset_73,subset_106' &
distillation 'subset_165,subset_91,subset_87,subset_142,subset_215,subset_130,subset_132,subset_5,subset_146,subset_191,subset_174' 'subset_83,subset_129,subset_113,subset_103,subset_117,subset_61' '39226'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_46,subset_137,subset_78,subset_23,subset_201,subset_96,subset_25,subset_106,subset_93,subset_178,subset_0' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_101,subset_10,subset_4,subset_63,subset_92,subset_97' &
distillation 'subset_38,subset_8,subset_190,subset_11,subset_73,subset_59,subset_40,subset_57,subset_86,subset_147,subset_74' 'subset_50,subset_0,subset_88,subset_119,subset_73,subset_106' '21292'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_71,subset_205,subset_141,subset_31,subset_2,subset_120,subset_10,subset_136,subset_129,subset_43,subset_197' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_120,subset_84,subset_71,subset_19,subset_58,subset_43' &
distillation 'subset_46,subset_137,subset_78,subset_23,subset_201,subset_96,subset_25,subset_106,subset_93,subset_178,subset_0' 'subset_101,subset_10,subset_4,subset_63,subset_92,subset_97' '73919'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_158,subset_199,subset_102,subset_65,subset_214,subset_95,subset_153,subset_192,subset_19,subset_156,subset_56' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_7,subset_3,subset_30,subset_29,subset_111,subset_136' &
distillation 'subset_71,subset_205,subset_141,subset_31,subset_2,subset_120,subset_10,subset_136,subset_129,subset_43,subset_197' 'subset_120,subset_84,subset_71,subset_19,subset_58,subset_43' '31519'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_124,subset_45,subset_159,subset_203,subset_166,subset_167,subset_61,subset_30,subset_188,subset_110,subset_207' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_46,subset_9,subset_40,subset_96,subset_70,subset_69' &
distillation 'subset_158,subset_199,subset_102,subset_65,subset_214,subset_95,subset_153,subset_192,subset_19,subset_156,subset_56' 'subset_7,subset_3,subset_30,subset_29,subset_111,subset_136' '74277'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_44,subset_183,subset_189,subset_177,subset_39,subset_216,subset_219,subset_63,subset_55,subset_173,subset_218' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_75,subset_55,subset_95,subset_57,subset_27,subset_41' &
distillation 'subset_124,subset_45,subset_159,subset_203,subset_166,subset_167,subset_61,subset_30,subset_188,subset_110,subset_207' 'subset_46,subset_9,subset_40,subset_96,subset_70,subset_69' '51007'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_67,subset_119,subset_140,subset_133,subset_182,subset_69,subset_52,subset_112,subset_28,subset_154,subset_169' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_37,subset_28,subset_12,subset_24,subset_62,subset_116' &
distillation 'subset_44,subset_183,subset_189,subset_177,subset_39,subset_216,subset_219,subset_63,subset_55,subset_173,subset_218' 'subset_75,subset_55,subset_95,subset_57,subset_27,subset_41' '88740'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_148,subset_20,subset_217,subset_180,subset_121,subset_17,subset_109,subset_79,subset_162,subset_118,subset_51' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_121,subset_102,subset_64,subset_86,subset_59,subset_115' &
distillation 'subset_67,subset_119,subset_140,subset_133,subset_182,subset_69,subset_52,subset_112,subset_28,subset_154,subset_169' 'subset_37,subset_28,subset_12,subset_24,subset_62,subset_116' '70852'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_171,subset_47,subset_135,subset_193,subset_101,subset_12,subset_168,subset_220,subset_97,subset_196,subset_16' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_93,subset_79,subset_105,subset_100,subset_81,subset_60' &
distillation 'subset_148,subset_20,subset_217,subset_180,subset_121,subset_17,subset_109,subset_79,subset_162,subset_118,subset_51' 'subset_121,subset_102,subset_64,subset_86,subset_59,subset_115' '43480'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_41,subset_76,subset_111,subset_157,subset_6,subset_184,subset_161,subset_22,subset_66,subset_7,subset_24' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_67,subset_131,subset_123,subset_18,subset_34,subset_54' &
distillation 'subset_171,subset_47,subset_135,subset_193,subset_101,subset_12,subset_168,subset_220,subset_97,subset_196,subset_16' 'subset_93,subset_79,subset_105,subset_100,subset_81,subset_60' '50311'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_117,subset_143,subset_127,subset_27,subset_113,subset_13,subset_1,subset_104,subset_176,subset_53,subset_145' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_77,subset_22,subset_68,subset_56,subset_17,subset_23' &
distillation 'subset_41,subset_76,subset_111,subset_157,subset_6,subset_184,subset_161,subset_22,subset_66,subset_7,subset_24' 'subset_67,subset_131,subset_123,subset_18,subset_34,subset_54' '99039'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_21,subset_139,subset_131,subset_222,subset_64,subset_9,subset_62,subset_14,subset_82,subset_4,subset_116' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_36,subset_13,subset_21,subset_112,subset_65,subset_90' &
distillation 'subset_117,subset_143,subset_127,subset_27,subset_113,subset_13,subset_1,subset_104,subset_176,subset_53,subset_145' 'subset_77,subset_22,subset_68,subset_56,subset_17,subset_23' '96959'
distillation 'subset_21,subset_139,subset_131,subset_222,subset_64,subset_9,subset_62,subset_14,subset_82,subset_4,subset_116' 'subset_36,subset_13,subset_21,subset_112,subset_65,subset_90' '96959'