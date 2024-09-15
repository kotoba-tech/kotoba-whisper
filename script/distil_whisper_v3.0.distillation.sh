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
    --logging_steps 50 \
    --attn_implementation "flash_attention_2" \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --output_dir "./${HF_MODEL_ALIAS}" \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --seed ${SEED} \
    --report_to "none" \
    --num_train_epochs 1
}

# Epoch 1
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_105,subset_195,subset_168,subset_52,subset_208' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_78,subset_134,subset_14' &
distillation 'subset_162,subset_36,subset_185,subset_2,subset_45' 'subset_67,subset_121,subset_126' '88039'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_212,subset_159,subset_110,subset_123,subset_121' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_113,subset_4' &
distillation 'subset_105,subset_195,subset_168,subset_52,subset_208' 'subset_78,subset_134,subset_14' '84607'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_98,subset_181,subset_60,subset_13,subset_44' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_26,subset_131,subset_38' &
distillation 'subset_212,subset_159,subset_110,subset_123,subset_121' 'subset_72,subset_113,subset_4' '39321'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_16,subset_84,subset_135,subset_29,subset_215' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_74,subset_40,subset_75' &
distillation 'subset_98,subset_181,subset_60,subset_13,subset_44' 'subset_26,subset_131,subset_38' '59929'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_21,subset_169,subset_194,subset_30,subset_127' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_98,subset_117,subset_22' &
distillation 'subset_16,subset_84,subset_135,subset_29,subset_215' 'subset_74,subset_40,subset_75' '41441'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_175,subset_4,subset_42,subset_47,subset_5' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_53,subset_47,subset_48' &
distillation 'subset_21,subset_169,subset_194,subset_30,subset_127' 'subset_98,subset_117,subset_22' '98548'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_77,subset_3,subset_15,subset_216,subset_190' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_64,subset_63,subset_25' &
distillation 'subset_175,subset_4,subset_42,subset_47,subset_5' 'subset_53,subset_47,subset_48' '9508'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_205,subset_81,subset_192,subset_99,subset_133' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_19,subset_28,subset_5' &
distillation 'subset_77,subset_3,subset_15,subset_216,subset_190' 'subset_64,subset_63,subset_25' '1220'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_66,subset_145,subset_73,subset_10,subset_122' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_107,subset_2,subset_81' &
distillation 'subset_205,subset_81,subset_192,subset_99,subset_133' 'subset_19,subset_28,subset_5' '60068'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_177,subset_9,subset_12,subset_38,subset_138' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_65,subset_36,subset_10' &
distillation 'subset_66,subset_145,subset_73,subset_10,subset_122' 'subset_107,subset_2,subset_81' '81416'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_61,subset_198,subset_89,subset_193,subset_134' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_33,subset_32,subset_127' &
distillation 'subset_177,subset_9,subset_12,subset_38,subset_138' 'subset_65,subset_36,subset_10' '73792'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_119,subset_79,subset_78,subset_209,subset_64' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_41,subset_39,subset_20' &
distillation 'subset_61,subset_198,subset_89,subset_193,subset_134' 'subset_33,subset_32,subset_127' '13104'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_103,subset_0,subset_111,subset_196,subset_180' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_49,subset_34,subset_37' &
distillation 'subset_119,subset_79,subset_78,subset_209,subset_64' 'subset_41,subset_39,subset_20' '9602'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_37,subset_222,subset_140,subset_126,subset_104' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_91,subset_106' &
distillation 'subset_103,subset_0,subset_111,subset_196,subset_180' 'subset_49,subset_34,subset_37' '70468'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_144,subset_187,subset_218,subset_32,subset_171' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_76,subset_79,subset_80' &
distillation 'subset_37,subset_222,subset_140,subset_126,subset_104' 'subset_3,subset_91,subset_106' '27938'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_152,subset_85,subset_48,subset_161,subset_125' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_105,subset_42,subset_97' &
distillation 'subset_144,subset_187,subset_218,subset_32,subset_171' 'subset_76,subset_79,subset_80' '66307'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_76,subset_100,subset_153,subset_128,subset_19' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_99,subset_46,subset_44' &
distillation 'subset_152,subset_85,subset_48,subset_161,subset_125' 'subset_105,subset_42,subset_97' '34760'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_131,subset_199,subset_156,subset_115,subset_65' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_66,subset_129,subset_29' &
distillation 'subset_76,subset_100,subset_153,subset_128,subset_19' 'subset_99,subset_46,subset_44' '17361'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_184,subset_221,subset_46,subset_51,subset_101' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_133,subset_50,subset_122' &
distillation 'subset_131,subset_199,subset_156,subset_115,subset_65' 'subset_66,subset_129,subset_29' '45745'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_54,subset_172,subset_106,subset_109,subset_142' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_87,subset_0,subset_88' &
distillation 'subset_184,subset_221,subset_46,subset_51,subset_101' 'subset_133,subset_50,subset_122' '9016'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_149,subset_95,subset_167,subset_157,subset_33' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_77,subset_95,subset_94' &
distillation 'subset_54,subset_172,subset_106,subset_109,subset_142' 'subset_87,subset_0,subset_88' '32018'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_164,subset_130,subset_82,subset_113,subset_197' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_124,subset_61,subset_125' &
distillation 'subset_149,subset_95,subset_167,subset_157,subset_33' 'subset_77,subset_95,subset_94' '48434'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_63,subset_132,subset_124,subset_186,subset_174' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_120,subset_21,subset_30' &
distillation 'subset_164,subset_130,subset_82,subset_113,subset_197' 'subset_124,subset_61,subset_125' '37353'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_112,subset_72,subset_120,subset_148,subset_27' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_11,subset_136,subset_123' &
distillation 'subset_63,subset_132,subset_124,subset_186,subset_174' 'subset_120,subset_21,subset_30' '20676'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_201,subset_34,subset_102,subset_80,subset_206' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_114,subset_96,subset_70' &
distillation 'subset_112,subset_72,subset_120,subset_148,subset_27' 'subset_11,subset_136,subset_123' '57433'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_155,subset_14,subset_83,subset_191,subset_69' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_137,subset_9,subset_90' &
distillation 'subset_201,subset_34,subset_102,subset_80,subset_206' 'subset_114,subset_96,subset_70' '71200'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_211,subset_118,subset_146,subset_217,subset_136' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_102,subset_35,subset_23' &
distillation 'subset_155,subset_14,subset_83,subset_191,subset_69' 'subset_137,subset_9,subset_90' '92214'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_43,subset_18,subset_68,subset_53,subset_90' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_104,subset_132,subset_57' &
distillation 'subset_211,subset_118,subset_146,subset_217,subset_136' 'subset_102,subset_35,subset_23' '39651'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_94,subset_41,subset_93,subset_116,subset_182' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_68,subset_89,subset_110' &
distillation 'subset_43,subset_18,subset_68,subset_53,subset_90' 'subset_104,subset_132,subset_57' '80173'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_176,subset_25,subset_202,subset_165,subset_74' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_115,subset_13,subset_43' &
distillation 'subset_94,subset_41,subset_93,subset_116,subset_182' 'subset_68,subset_89,subset_110' '85717'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_58,subset_170,subset_17,subset_49,subset_147' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_101,subset_51,subset_7' &
distillation 'subset_176,subset_25,subset_202,subset_165,subset_74' 'subset_115,subset_13,subset_43' '69329'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_92,subset_158,subset_160,subset_75,subset_141' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_108,subset_82,subset_83' &
distillation 'subset_58,subset_170,subset_17,subset_49,subset_147' 'subset_101,subset_51,subset_7' '1025'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_20,subset_96,subset_31,subset_137,subset_117' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_86,subset_6,subset_93' &
distillation 'subset_92,subset_158,subset_160,subset_75,subset_141' 'subset_108,subset_82,subset_83' '87538'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_11,subset_67,subset_200,subset_88,subset_91' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_59,subset_111,subset_54' &
distillation 'subset_20,subset_96,subset_31,subset_137,subset_117' 'subset_86,subset_6,subset_93' '72692'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_24,subset_97,subset_204,subset_213,subset_86' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_45,subset_55,subset_130' &
distillation 'subset_11,subset_67,subset_200,subset_88,subset_91' 'subset_59,subset_111,subset_54' '39240'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_203,subset_39,subset_214,subset_87,subset_207' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_12,subset_24' &
distillation 'subset_24,subset_97,subset_204,subset_213,subset_86' 'subset_45,subset_55,subset_130' '86951'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_178,subset_40,subset_1,subset_71,subset_150' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_52,subset_103,subset_116' &
distillation 'subset_203,subset_39,subset_214,subset_87,subset_207' 'subset_109,subset_12,subset_24' '13577'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_114,subset_56,subset_107,subset_210,subset_179' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_100,subset_31,subset_128' &
distillation 'subset_178,subset_40,subset_1,subset_71,subset_150' 'subset_52,subset_103,subset_116' '17601'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_166,subset_183,subset_50,subset_143,subset_220' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_118,subset_73,subset_112' &
distillation 'subset_114,subset_56,subset_107,subset_210,subset_179' 'subset_100,subset_31,subset_128' '34664'
distillation 'subset_166,subset_183,subset_50,subset_143,subset_220' 'subset_118,subset_73,subset_112' '34664'
# Epoch 2
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_14,subset_5,subset_134,subset_83,subset_220' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_10,subset_80,subset_74' &
distillation 'subset_166,subset_183,subset_50,subset_143,subset_220' 'subset_118,subset_73,subset_112' '14168'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_72,subset_39,subset_191,subset_85,subset_2' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_30,subset_133,subset_75' &
distillation 'subset_14,subset_5,subset_134,subset_83,subset_220' 'subset_10,subset_80,subset_74' '75849'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_138,subset_41,subset_202,subset_144,subset_221' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_53,subset_33,subset_27' &
distillation 'subset_72,subset_39,subset_191,subset_85,subset_2' 'subset_30,subset_133,subset_75' '3365'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_179,subset_174,subset_102,subset_55,subset_51' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_106,subset_15,subset_66' &
distillation 'subset_138,subset_41,subset_202,subset_144,subset_221' 'subset_53,subset_33,subset_27' '40888'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_77,subset_217,subset_58,subset_43,subset_57' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_45,subset_98,subset_87' &
distillation 'subset_179,subset_174,subset_102,subset_55,subset_51' 'subset_106,subset_15,subset_66' '75470'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_71,subset_63,subset_95,subset_213,subset_79' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_119,subset_134,subset_63' &
distillation 'subset_77,subset_217,subset_58,subset_43,subset_57' 'subset_45,subset_98,subset_87' '88781'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_4,subset_92,subset_40,subset_36,subset_93' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_132,subset_62' &
distillation 'subset_71,subset_63,subset_95,subset_213,subset_79' 'subset_119,subset_134,subset_63' '49192'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_110,subset_129,subset_207,subset_161,subset_117' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_35,subset_114,subset_104' &
distillation 'subset_4,subset_92,subset_40,subset_36,subset_93' 'subset_72,subset_132,subset_62' '51990'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_84,subset_152,subset_128,subset_167,subset_100' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_137,subset_110,subset_9' &
distillation 'subset_110,subset_129,subset_207,subset_161,subset_117' 'subset_35,subset_114,subset_104' '93718'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_13,subset_109,subset_116,subset_11,subset_73' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_95,subset_60,subset_25' &
distillation 'subset_84,subset_152,subset_128,subset_167,subset_100' 'subset_137,subset_110,subset_9' '25995'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_78,subset_175,subset_166,subset_186,subset_99' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_118,subset_24,subset_97' &
distillation 'subset_13,subset_109,subset_116,subset_11,subset_73' 'subset_95,subset_60,subset_25' '9961'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_94,subset_178,subset_163,subset_16,subset_204' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_94,subset_6,subset_54' &
distillation 'subset_78,subset_175,subset_166,subset_186,subset_99' 'subset_118,subset_24,subset_97' '77607'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_101,subset_141,subset_177,subset_108,subset_96' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_51,subset_77,subset_112' &
distillation 'subset_94,subset_178,subset_163,subset_16,subset_204' 'subset_94,subset_6,subset_54' '90520'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_3,subset_62,subset_56,subset_139,subset_48' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_79,subset_40,subset_48' &
distillation 'subset_101,subset_141,subset_177,subset_108,subset_96' 'subset_51,subset_77,subset_112' '82213'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_120,subset_10,subset_137,subset_165,subset_181' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_19,subset_117,subset_61' &
distillation 'subset_3,subset_62,subset_56,subset_139,subset_48' 'subset_79,subset_40,subset_48' '31830'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_154,subset_26,subset_90,subset_147,subset_219' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_38,subset_69,subset_103' &
distillation 'subset_120,subset_10,subset_137,subset_165,subset_181' 'subset_19,subset_117,subset_61' '13356'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_158,subset_7,subset_214,subset_98,subset_80' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_82,subset_37,subset_14' &
distillation 'subset_154,subset_26,subset_90,subset_147,subset_219' 'subset_38,subset_69,subset_103' '91382'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_8,subset_171,subset_192,subset_197,subset_52' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_8,subset_101,subset_52' &
distillation 'subset_158,subset_7,subset_214,subset_98,subset_80' 'subset_82,subset_37,subset_14' '39529'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_172,subset_145,subset_21,subset_35,subset_184' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_39,subset_28,subset_121' &
distillation 'subset_8,subset_171,subset_192,subset_197,subset_52' 'subset_8,subset_101,subset_52' '89687'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_86,subset_156,subset_173,subset_25,subset_0' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_21,subset_49,subset_85' &
distillation 'subset_172,subset_145,subset_21,subset_35,subset_184' 'subset_39,subset_28,subset_121' '78697'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_143,subset_130,subset_20,subset_105,subset_91' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_50,subset_59,subset_96' &
distillation 'subset_86,subset_156,subset_173,subset_25,subset_0' 'subset_21,subset_49,subset_85' '15866'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_215,subset_126,subset_146,subset_34,subset_103' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_64,subset_99,subset_91' &
distillation 'subset_143,subset_130,subset_20,subset_105,subset_91' 'subset_50,subset_59,subset_96' '74177'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_218,subset_132,subset_176,subset_54,subset_67' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_70,subset_78,subset_11' &
distillation 'subset_215,subset_126,subset_146,subset_34,subset_103' 'subset_64,subset_99,subset_91' '5383'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_133,subset_61,subset_187,subset_23,subset_45' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_17,subset_23' &
distillation 'subset_218,subset_132,subset_176,subset_54,subset_67' 'subset_70,subset_78,subset_11' '45508'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_148,subset_104,subset_135,subset_183,subset_27' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_56,subset_2,subset_43' &
distillation 'subset_133,subset_61,subset_187,subset_23,subset_45' 'subset_129,subset_17,subset_23' '69827'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_82,subset_121,subset_168,subset_211,subset_118' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_125,subset_68,subset_57' &
distillation 'subset_148,subset_104,subset_135,subset_183,subset_27' 'subset_56,subset_2,subset_43' '56148'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_155,subset_68,subset_69,subset_60,subset_222' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_20,subset_76,subset_67' &
distillation 'subset_82,subset_121,subset_168,subset_211,subset_118' 'subset_125,subset_68,subset_57' '86706'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_125,subset_106,subset_122,subset_170,subset_89' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_0,subset_105,subset_136' &
distillation 'subset_155,subset_68,subset_69,subset_60,subset_222' 'subset_20,subset_76,subset_67' '48571'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_193,subset_200,subset_19,subset_124,subset_9' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_126,subset_123,subset_93' &
distillation 'subset_125,subset_106,subset_122,subset_170,subset_89' 'subset_0,subset_105,subset_136' '9038'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_198,subset_112,subset_107,subset_46,subset_28' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_58,subset_5,subset_113' &
distillation 'subset_193,subset_200,subset_19,subset_124,subset_9' 'subset_126,subset_123,subset_93' '66317'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_205,subset_196,subset_31,subset_17,subset_50' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_7,subset_122,subset_55' &
distillation 'subset_198,subset_112,subset_107,subset_46,subset_28' 'subset_58,subset_5,subset_113' '84874'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_59,subset_195,subset_149,subset_66,subset_210' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_46,subset_44' &
distillation 'subset_205,subset_196,subset_31,subset_17,subset_50' 'subset_7,subset_122,subset_55' '44725'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_119,subset_70,subset_49,subset_142,subset_1' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_4,subset_107,subset_115' &
distillation 'subset_59,subset_195,subset_149,subset_66,subset_210' 'subset_109,subset_46,subset_44' '1658'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_131,subset_169,subset_209,subset_188,subset_199' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_111,subset_89,subset_42' &
distillation 'subset_119,subset_70,subset_49,subset_142,subset_1' 'subset_4,subset_107,subset_115' '55057'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_88,subset_115,subset_32,subset_212,subset_42' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_81,subset_47,subset_22' &
distillation 'subset_131,subset_169,subset_209,subset_188,subset_199' 'subset_111,subset_89,subset_42' '64251'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_164,subset_15,subset_6,subset_182,subset_65' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_131,subset_120,subset_84' &
distillation 'subset_88,subset_115,subset_32,subset_212,subset_42' 'subset_81,subset_47,subset_22' '13833'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_208,subset_136,subset_216,subset_97,subset_44' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_32,subset_128' &
distillation 'subset_164,subset_15,subset_6,subset_182,subset_65' 'subset_131,subset_120,subset_84' '56822'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_30,subset_185,subset_53,subset_33,subset_203' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_41,subset_13,subset_26' &
distillation 'subset_208,subset_136,subset_216,subset_97,subset_44' 'subset_3,subset_32,subset_128' '47472'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_189,subset_113,subset_150,subset_81,subset_151' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_29,subset_16,subset_65' &
distillation 'subset_30,subset_185,subset_53,subset_33,subset_203' 'subset_41,subset_13,subset_26' '83307'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_75,subset_47,subset_76,subset_162,subset_159' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_90,subset_88,subset_71' &
distillation 'subset_189,subset_113,subset_150,subset_81,subset_151' 'subset_29,subset_16,subset_65' '60258'
distillation 'subset_75,subset_47,subset_76,subset_162,subset_159' 'subset_90,subset_88,subset_71' '60258'
# Epoch 3
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_108,subset_0,subset_166,subset_211,subset_152' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_68,subset_14,subset_21' &
distillation 'subset_75,subset_47,subset_76,subset_162,subset_159' 'subset_90,subset_88,subset_71' '13069'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_103,subset_187,subset_25,subset_20,subset_17' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_117,subset_88,subset_116' &
distillation 'subset_108,subset_0,subset_166,subset_211,subset_152' 'subset_68,subset_14,subset_21' '68937'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_37,subset_181,subset_57,subset_170,subset_126' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_114,subset_110,subset_74' &
distillation 'subset_103,subset_187,subset_25,subset_20,subset_17' 'subset_117,subset_88,subset_116' '59890'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_75,subset_176,subset_145,subset_189,subset_42' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_2,subset_43,subset_0' &
distillation 'subset_37,subset_181,subset_57,subset_170,subset_126' 'subset_114,subset_110,subset_74' '2011'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_83,subset_215,subset_168,subset_33,subset_130' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_94,subset_75,subset_58' &
distillation 'subset_75,subset_176,subset_145,subset_189,subset_42' 'subset_2,subset_43,subset_0' '94692'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_222,subset_203,subset_68,subset_160,subset_87' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_83,subset_136,subset_31' &
distillation 'subset_83,subset_215,subset_168,subset_33,subset_130' 'subset_94,subset_75,subset_58' '18894'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_78,subset_3,subset_102,subset_35,subset_128' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_37,subset_33,subset_127' &
distillation 'subset_222,subset_203,subset_68,subset_160,subset_87' 'subset_83,subset_136,subset_31' '53736'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_93,subset_112,subset_47,subset_135,subset_107' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_95,subset_122,subset_7' &
distillation 'subset_78,subset_3,subset_102,subset_35,subset_128' 'subset_37,subset_33,subset_127' '85823'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_84,subset_209,subset_155,subset_124,subset_79' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_131,subset_103,subset_4' &
distillation 'subset_93,subset_112,subset_47,subset_135,subset_107' 'subset_95,subset_122,subset_7' '20232'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_54,subset_118,subset_18,subset_86,subset_81' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_73,subset_11,subset_89' &
distillation 'subset_84,subset_209,subset_155,subset_124,subset_79' 'subset_131,subset_103,subset_4' '9807'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_6,subset_55,subset_89,subset_12,subset_100' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_77,subset_78,subset_82' &
distillation 'subset_54,subset_118,subset_18,subset_86,subset_81' 'subset_73,subset_11,subset_89' '61537'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_219,subset_56,subset_14,subset_9,subset_1' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_20,subset_65,subset_59' &
distillation 'subset_6,subset_55,subset_89,subset_12,subset_100' 'subset_77,subset_78,subset_82' '34737'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_148,subset_41,subset_123,subset_77,subset_28' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_42,subset_47,subset_108' &
distillation 'subset_219,subset_56,subset_14,subset_9,subset_1' 'subset_20,subset_65,subset_59' '44375'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_36,subset_95,subset_63,subset_202,subset_113' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_35,subset_28,subset_113' &
distillation 'subset_148,subset_41,subset_123,subset_77,subset_28' 'subset_42,subset_47,subset_108' '81691'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_194,subset_185,subset_8,subset_119,subset_163' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_27,subset_84' &
distillation 'subset_36,subset_95,subset_63,subset_202,subset_113' 'subset_35,subset_28,subset_113' '90794'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_164,subset_44,subset_213,subset_26,subset_90' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_112,subset_45,subset_55' &
distillation 'subset_194,subset_185,subset_8,subset_119,subset_163' 'subset_3,subset_27,subset_84' '52098'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_186,subset_216,subset_179,subset_92,subset_138' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_106,subset_54,subset_25' &
distillation 'subset_164,subset_44,subset_213,subset_26,subset_90' 'subset_112,subset_45,subset_55' '85175'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_140,subset_15,subset_127,subset_51,subset_143' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_99,subset_39,subset_125' &
distillation 'subset_186,subset_216,subset_179,subset_92,subset_138' 'subset_106,subset_54,subset_25' '10527'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_97,subset_205,subset_146,subset_2,subset_218' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_52,subset_128,subset_63' &
distillation 'subset_140,subset_15,subset_127,subset_51,subset_143' 'subset_99,subset_39,subset_125' '43065'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_175,subset_85,subset_221,subset_32,subset_40' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_50,subset_34,subset_72' &
distillation 'subset_97,subset_205,subset_146,subset_2,subset_218' 'subset_52,subset_128,subset_63' '88389'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_76,subset_142,subset_167,subset_70,subset_199' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_38,subset_111,subset_24' &
distillation 'subset_175,subset_85,subset_221,subset_32,subset_40' 'subset_50,subset_34,subset_72' '69942'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_101,subset_129,subset_165,subset_206,subset_144' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_132,subset_57,subset_44' &
distillation 'subset_76,subset_142,subset_167,subset_70,subset_199' 'subset_38,subset_111,subset_24' '49808'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_210,subset_156,subset_111,subset_7,subset_59' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_81,subset_60,subset_120' &
distillation 'subset_101,subset_129,subset_165,subset_206,subset_144' 'subset_132,subset_57,subset_44' '41505'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_52,subset_207,subset_180,subset_39,subset_72' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_46,subset_56,subset_80' &
distillation 'subset_210,subset_156,subset_111,subset_7,subset_59' 'subset_81,subset_60,subset_120' '82155'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_110,subset_174,subset_64,subset_38,subset_5' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_105,subset_79,subset_71' &
distillation 'subset_52,subset_207,subset_180,subset_39,subset_72' 'subset_46,subset_56,subset_80' '94203'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_161,subset_22,subset_21,subset_16,subset_49' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_118,subset_30,subset_119' &
distillation 'subset_110,subset_174,subset_64,subset_38,subset_5' 'subset_105,subset_79,subset_71' '99536'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_106,subset_71,subset_137,subset_208,subset_24' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_96,subset_93,subset_29' &
distillation 'subset_161,subset_22,subset_21,subset_16,subset_49' 'subset_118,subset_30,subset_119' '63956'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_94,subset_50,subset_116,subset_153,subset_88' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_134,subset_124,subset_6' &
distillation 'subset_106,subset_71,subset_137,subset_208,subset_24' 'subset_96,subset_93,subset_29' '70912'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_158,subset_73,subset_4,subset_204,subset_96' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_9,subset_109,subset_121' &
distillation 'subset_94,subset_50,subset_116,subset_153,subset_88' 'subset_134,subset_124,subset_6' '4705'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_66,subset_74,subset_61,subset_30,subset_80' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_67,subset_16,subset_76' &
distillation 'subset_158,subset_73,subset_4,subset_204,subset_96' 'subset_9,subset_109,subset_121' '80931'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_193,subset_34,subset_197,subset_125,subset_200' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_100,subset_70,subset_137' &
distillation 'subset_66,subset_74,subset_61,subset_30,subset_80' 'subset_67,subset_16,subset_76' '8968'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_188,subset_184,subset_46,subset_147,subset_159' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_53,subset_90,subset_85' &
distillation 'subset_193,subset_34,subset_197,subset_125,subset_200' 'subset_100,subset_70,subset_137' '30771'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_105,subset_178,subset_154,subset_43,subset_220' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_135,subset_10' &
distillation 'subset_188,subset_184,subset_46,subset_147,subset_159' 'subset_53,subset_90,subset_85' '82727'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_67,subset_151,subset_131,subset_104,subset_212' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_91,subset_13,subset_115' &
distillation 'subset_105,subset_178,subset_154,subset_43,subset_220' 'subset_129,subset_135,subset_10' '89705'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_182,subset_82,subset_91,subset_141,subset_217' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_87,subset_1,subset_97' &
distillation 'subset_67,subset_151,subset_131,subset_104,subset_212' 'subset_91,subset_13,subset_115' '37667'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_11,subset_69,subset_134,subset_19,subset_120' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_40,subset_66,subset_49' &
distillation 'subset_182,subset_82,subset_91,subset_141,subset_217' 'subset_87,subset_1,subset_97' '29817'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_65,subset_31,subset_157,subset_192,subset_62' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_133,subset_17,subset_107' &
distillation 'subset_11,subset_69,subset_134,subset_19,subset_120' 'subset_40,subset_66,subset_49' '97880'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_172,subset_99,subset_27,subset_109,subset_98' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_22,subset_8,subset_126' &
distillation 'subset_65,subset_31,subset_157,subset_192,subset_62' 'subset_133,subset_17,subset_107' '11846'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_121,subset_191,subset_171,subset_173,subset_115' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_51,subset_64,subset_5' &
distillation 'subset_172,subset_99,subset_27,subset_109,subset_98' 'subset_22,subset_8,subset_126' '56881'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_195,subset_198,subset_23,subset_150,subset_29' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_92,subset_130,subset_62' &
distillation 'subset_121,subset_191,subset_171,subset_173,subset_115' 'subset_51,subset_64,subset_5' '12903'
distillation 'subset_195,subset_198,subset_23,subset_150,subset_29' 'subset_92,subset_130,subset_62' '12903'
# Epoch 4
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_199,subset_149,subset_195,subset_109,subset_174' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_31,subset_78,subset_19' &
distillation 'subset_195,subset_198,subset_23,subset_150,subset_29' 'subset_92,subset_130,subset_62' '3794'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_11,subset_95,subset_140,subset_52,subset_22' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_68,subset_79,subset_22' &
distillation 'subset_199,subset_149,subset_195,subset_109,subset_174' 'subset_31,subset_78,subset_19' '23647'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_178,subset_106,subset_14,subset_42,subset_78' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_123,subset_82,subset_120' &
distillation 'subset_11,subset_95,subset_140,subset_52,subset_22' 'subset_68,subset_79,subset_22' '35818'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_153,subset_13,subset_108,subset_168,subset_157' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_72,subset_135,subset_104' &
distillation 'subset_178,subset_106,subset_14,subset_42,subset_78' 'subset_123,subset_82,subset_120' '92104'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_110,subset_67,subset_100,subset_143,subset_30' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_80,subset_121,subset_126' &
distillation 'subset_153,subset_13,subset_108,subset_168,subset_157' 'subset_72,subset_135,subset_104' '99930'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_24,subset_26,subset_198,subset_34,subset_196' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_83,subset_112,subset_28' &
distillation 'subset_110,subset_67,subset_100,subset_143,subset_30' 'subset_80,subset_121,subset_126' '40499'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_120,subset_144,subset_81,subset_169,subset_158' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_25,subset_134,subset_42' &
distillation 'subset_24,subset_26,subset_198,subset_34,subset_196' 'subset_83,subset_112,subset_28' '44541'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_138,subset_186,subset_219,subset_82,subset_46' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_0,subset_97,subset_33' &
distillation 'subset_120,subset_144,subset_81,subset_169,subset_158' 'subset_25,subset_134,subset_42' '46009'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_54,subset_107,subset_3,subset_214,subset_71' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_64,subset_14,subset_35' &
distillation 'subset_138,subset_186,subset_219,subset_82,subset_46' 'subset_0,subset_97,subset_33' '800'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_218,subset_59,subset_117,subset_12,subset_156' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_16,subset_131,subset_65' &
distillation 'subset_54,subset_107,subset_3,subset_214,subset_71' 'subset_64,subset_14,subset_35' '23777'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_203,subset_8,subset_16,subset_179,subset_60' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_133,subset_48,subset_30' &
distillation 'subset_218,subset_59,subset_117,subset_12,subset_156' 'subset_16,subset_131,subset_65' '18776'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_137,subset_116,subset_69,subset_173,subset_99' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_115,subset_137,subset_105' &
distillation 'subset_203,subset_8,subset_16,subset_179,subset_60' 'subset_133,subset_48,subset_30' '74228'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_204,subset_98,subset_187,subset_5,subset_77' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_62,subset_1,subset_29' &
distillation 'subset_137,subset_116,subset_69,subset_173,subset_99' 'subset_115,subset_137,subset_105' '86159'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_31,subset_152,subset_125,subset_145,subset_165' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_54,subset_17,subset_130' &
distillation 'subset_204,subset_98,subset_187,subset_5,subset_77' 'subset_62,subset_1,subset_29' '52531'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_89,subset_135,subset_192,subset_162,subset_104' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_55,subset_26,subset_87' &
distillation 'subset_31,subset_152,subset_125,subset_145,subset_165' 'subset_54,subset_17,subset_130' '9120'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_209,subset_159,subset_211,subset_105,subset_87' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_88,subset_77,subset_27' &
distillation 'subset_89,subset_135,subset_192,subset_162,subset_104' 'subset_55,subset_26,subset_87' '18595'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_170,subset_91,subset_216,subset_4,subset_163' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_7,subset_93' &
distillation 'subset_209,subset_159,subset_211,subset_105,subset_87' 'subset_88,subset_77,subset_27' '97120'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_161,subset_127,subset_40,subset_18,subset_66' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_100,subset_10' &
distillation 'subset_170,subset_91,subset_216,subset_4,subset_163' 'subset_109,subset_7,subset_93' '82991'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_75,subset_202,subset_208,subset_0,subset_212' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_127,subset_98,subset_107' &
distillation 'subset_161,subset_127,subset_40,subset_18,subset_66' 'subset_3,subset_100,subset_10' '4018'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_201,subset_64,subset_185,subset_220,subset_197' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_76,subset_103,subset_132' &
distillation 'subset_75,subset_202,subset_208,subset_0,subset_212' 'subset_127,subset_98,subset_107' '12026'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_146,subset_15,subset_160,subset_85,subset_171' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_39,subset_95,subset_4' &
distillation 'subset_201,subset_64,subset_185,subset_220,subset_197' 'subset_76,subset_103,subset_132' '97794'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_17,subset_27,subset_183,subset_167,subset_150' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_89,subset_66,subset_46' &
distillation 'subset_146,subset_15,subset_160,subset_85,subset_171' 'subset_39,subset_95,subset_4' '69531'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_190,subset_139,subset_70,subset_191,subset_58' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_73,subset_5,subset_94' &
distillation 'subset_17,subset_27,subset_183,subset_167,subset_150' 'subset_89,subset_66,subset_46' '28197'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_102,subset_215,subset_49,subset_128,subset_151' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_128,subset_40,subset_129' &
distillation 'subset_190,subset_139,subset_70,subset_191,subset_58' 'subset_73,subset_5,subset_94' '49308'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_73,subset_63,subset_29,subset_50,subset_113' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_61,subset_92,subset_106' &
distillation 'subset_102,subset_215,subset_49,subset_128,subset_151' 'subset_128,subset_40,subset_129' '55028'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_76,subset_176,subset_55,subset_23,subset_129' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_6,subset_15,subset_71' &
distillation 'subset_73,subset_63,subset_29,subset_50,subset_113' 'subset_61,subset_92,subset_106' '59460'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_122,subset_200,subset_33,subset_35,subset_2' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_59,subset_32,subset_51' &
distillation 'subset_76,subset_176,subset_55,subset_23,subset_129' 'subset_6,subset_15,subset_71' '44681'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_20,subset_111,subset_90,subset_37,subset_130' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_37,subset_119,subset_101' &
distillation 'subset_122,subset_200,subset_33,subset_35,subset_2' 'subset_59,subset_32,subset_51' '20631'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_93,subset_131,subset_86,subset_56,subset_97' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_43,subset_2,subset_63' &
distillation 'subset_20,subset_111,subset_90,subset_37,subset_130' 'subset_37,subset_119,subset_101' '48503'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_9,subset_19,subset_121,subset_57,subset_10' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_24,subset_49,subset_67' &
distillation 'subset_93,subset_131,subset_86,subset_56,subset_97' 'subset_43,subset_2,subset_63' '40846'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_132,subset_194,subset_142,subset_103,subset_188' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_52,subset_99,subset_111' &
distillation 'subset_9,subset_19,subset_121,subset_57,subset_10' 'subset_24,subset_49,subset_67' '94581'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_45,subset_61,subset_114,subset_74,subset_136' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_110,subset_36,subset_124' &
distillation 'subset_132,subset_194,subset_142,subset_103,subset_188' 'subset_52,subset_99,subset_111' '42517'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_148,subset_68,subset_38,subset_32,subset_6' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_34,subset_108,subset_81' &
distillation 'subset_45,subset_61,subset_114,subset_74,subset_136' 'subset_110,subset_36,subset_124' '74395'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_213,subset_207,subset_28,subset_206,subset_172' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_9,subset_13,subset_53' &
distillation 'subset_148,subset_68,subset_38,subset_32,subset_6' 'subset_34,subset_108,subset_81' '78148'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_48,subset_193,subset_51,subset_53,subset_189' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_116,subset_96,subset_23' &
distillation 'subset_213,subset_207,subset_28,subset_206,subset_172' 'subset_9,subset_13,subset_53' '11135'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_84,subset_88,subset_141,subset_119,subset_72' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_12,subset_8,subset_75' &
distillation 'subset_48,subset_193,subset_51,subset_53,subset_189' 'subset_116,subset_96,subset_23' '6895'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_166,subset_182,subset_62,subset_180,subset_36' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_18,subset_44,subset_45' &
distillation 'subset_84,subset_88,subset_141,subset_119,subset_72' 'subset_12,subset_8,subset_75' '20392'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_43,subset_39,subset_1,subset_112,subset_80' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_91,subset_50,subset_102' &
distillation 'subset_166,subset_182,subset_62,subset_180,subset_36' 'subset_18,subset_44,subset_45' '20632'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_21,subset_41,subset_133,subset_147,subset_221' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_74,subset_85,subset_58' &
distillation 'subset_43,subset_39,subset_1,subset_112,subset_80' 'subset_91,subset_50,subset_102' '98885'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_184,subset_83,subset_79,subset_124,subset_118' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_60,subset_125,subset_90' &
distillation 'subset_21,subset_41,subset_133,subset_147,subset_221' 'subset_74,subset_85,subset_58' '80981'
distillation 'subset_184,subset_83,subset_79,subset_124,subset_118' 'subset_60,subset_125,subset_90' '80981'
# Epoch 5
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_41,subset_97,subset_199,subset_79,subset_73' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_34,subset_38,subset_85' &
distillation 'subset_184,subset_83,subset_79,subset_124,subset_118' 'subset_60,subset_125,subset_90' '31252'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_108,subset_29,subset_182,subset_76,subset_203' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_130,subset_82,subset_86' &
distillation 'subset_41,subset_97,subset_199,subset_79,subset_73' 'subset_34,subset_38,subset_85' '55027'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_126,subset_83,subset_1,subset_33,subset_13' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_57,subset_29,subset_17' &
distillation 'subset_108,subset_29,subset_182,subset_76,subset_203' 'subset_130,subset_82,subset_86' '83748'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_58,subset_131,subset_115,subset_176,subset_80' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_84,subset_137,subset_104' &
distillation 'subset_126,subset_83,subset_1,subset_33,subset_13' 'subset_57,subset_29,subset_17' '81633'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_144,subset_183,subset_152,subset_20,subset_158' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_45,subset_48,subset_73' &
distillation 'subset_58,subset_131,subset_115,subset_176,subset_80' 'subset_84,subset_137,subset_104' '59914'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_43,subset_162,subset_136,subset_189,subset_212' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_44,subset_115,subset_1' &
distillation 'subset_144,subset_183,subset_152,subset_20,subset_158' 'subset_45,subset_48,subset_73' '8232'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_159,subset_119,subset_98,subset_96,subset_202' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_88,subset_31,subset_75' &
distillation 'subset_43,subset_162,subset_136,subset_189,subset_212' 'subset_44,subset_115,subset_1' '14530'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_166,subset_109,subset_188,subset_26,subset_145' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_2,subset_3,subset_20' &
distillation 'subset_159,subset_119,subset_98,subset_96,subset_202' 'subset_88,subset_31,subset_75' '65513'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_118,subset_3,subset_143,subset_28,subset_215' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_55,subset_58,subset_124' &
distillation 'subset_166,subset_109,subset_188,subset_26,subset_145' 'subset_2,subset_3,subset_20' '78215'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_59,subset_141,subset_12,subset_9,subset_10' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_93,subset_74,subset_4' &
distillation 'subset_118,subset_3,subset_143,subset_28,subset_215' 'subset_55,subset_58,subset_124' '70261'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_153,subset_195,subset_210,subset_125,subset_50' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_23,subset_25,subset_19' &
distillation 'subset_59,subset_141,subset_12,subset_9,subset_10' 'subset_93,subset_74,subset_4' '2154'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_135,subset_37,subset_67,subset_179,subset_177' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_120,subset_51,subset_53' &
distillation 'subset_153,subset_195,subset_210,subset_125,subset_50' 'subset_23,subset_25,subset_19' '82831'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_8,subset_95,subset_123,subset_146,subset_148' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_69,subset_37,subset_11' &
distillation 'subset_135,subset_37,subset_67,subset_179,subset_177' 'subset_120,subset_51,subset_53' '67551'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_114,subset_0,subset_197,subset_30,subset_93' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_108,subset_87,subset_98' &
distillation 'subset_8,subset_95,subset_123,subset_146,subset_148' 'subset_69,subset_37,subset_11' '75372'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_133,subset_89,subset_154,subset_205,subset_84' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_101,subset_46,subset_12' &
distillation 'subset_114,subset_0,subset_197,subset_30,subset_93' 'subset_108,subset_87,subset_98' '31711'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_184,subset_163,subset_186,subset_117,subset_171' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_103,subset_61,subset_68' &
distillation 'subset_133,subset_89,subset_154,subset_205,subset_84' 'subset_101,subset_46,subset_12' '94181'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_213,subset_17,subset_72,subset_46,subset_102' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_35,subset_78,subset_97' &
distillation 'subset_184,subset_163,subset_186,subset_117,subset_171' 'subset_103,subset_61,subset_68' '18818'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_77,subset_218,subset_140,subset_209,subset_39' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_116,subset_126,subset_96' &
distillation 'subset_213,subset_17,subset_72,subset_46,subset_102' 'subset_35,subset_78,subset_97' '38177'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_82,subset_181,subset_201,subset_204,subset_52' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_16,subset_106,subset_72' &
distillation 'subset_77,subset_218,subset_140,subset_209,subset_39' 'subset_116,subset_126,subset_96' '56261'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_90,subset_151,subset_54,subset_70,subset_107' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_40,subset_67,subset_131' &
distillation 'subset_82,subset_181,subset_201,subset_204,subset_52' 'subset_16,subset_106,subset_72' '205'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_130,subset_91,subset_216,subset_134,subset_40' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_89,subset_71' &
distillation 'subset_90,subset_151,subset_54,subset_70,subset_107' 'subset_40,subset_67,subset_131' '80584'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_69,subset_61,subset_132,subset_94,subset_64' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_135,subset_99,subset_83' &
distillation 'subset_130,subset_91,subset_216,subset_134,subset_40' 'subset_129,subset_89,subset_71' '46244'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_99,subset_5,subset_38,subset_220,subset_63' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_121,subset_119,subset_95' &
distillation 'subset_69,subset_61,subset_132,subset_94,subset_64' 'subset_135,subset_99,subset_83' '31531'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_22,subset_142,subset_128,subset_88,subset_86' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_123,subset_30,subset_28' &
distillation 'subset_99,subset_5,subset_38,subset_220,subset_63' 'subset_121,subset_119,subset_95' '74798'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_112,subset_57,subset_24,subset_81,subset_155' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_14,subset_136,subset_10' &
distillation 'subset_22,subset_142,subset_128,subset_88,subset_86' 'subset_123,subset_30,subset_28' '54600'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_42,subset_62,subset_219,subset_129,subset_66' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_60,subset_122,subset_118' &
distillation 'subset_112,subset_57,subset_24,subset_81,subset_155' 'subset_14,subset_136,subset_10' '24551'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_156,subset_35,subset_25,subset_122,subset_32' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_52,subset_132,subset_90' &
distillation 'subset_42,subset_62,subset_219,subset_129,subset_66' 'subset_60,subset_122,subset_118' '87080'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_217,subset_161,subset_221,subset_127,subset_87' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_92,subset_27,subset_64' &
distillation 'subset_156,subset_35,subset_25,subset_122,subset_32' 'subset_52,subset_132,subset_90' '87584'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_206,subset_65,subset_124,subset_160,subset_68' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_39,subset_117,subset_110' &
distillation 'subset_217,subset_161,subset_221,subset_127,subset_87' 'subset_92,subset_27,subset_64' '11221'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_187,subset_198,subset_207,subset_100,subset_208' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_94,subset_65,subset_114' &
distillation 'subset_206,subset_65,subset_124,subset_160,subset_68' 'subset_39,subset_117,subset_110' '68614'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_6,subset_214,subset_180,subset_11,subset_74' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_62,subset_59,subset_9' &
distillation 'subset_187,subset_198,subset_207,subset_100,subset_208' 'subset_94,subset_65,subset_114' '47265'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_175,subset_36,subset_92,subset_85,subset_19' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_133,subset_113,subset_63' &
distillation 'subset_6,subset_214,subset_180,subset_11,subset_74' 'subset_62,subset_59,subset_9' '8870'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_150,subset_121,subset_170,subset_157,subset_49' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_47,subset_100,subset_22' &
distillation 'subset_175,subset_36,subset_92,subset_85,subset_19' 'subset_133,subset_113,subset_63' '68951'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_174,subset_139,subset_178,subset_164,subset_168' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_107,subset_111,subset_0' &
distillation 'subset_150,subset_121,subset_170,subset_157,subset_49' 'subset_47,subset_100,subset_22' '71313'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_75,subset_193,subset_56,subset_23,subset_173' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_91,subset_125,subset_49' &
distillation 'subset_174,subset_139,subset_178,subset_164,subset_168' 'subset_107,subset_111,subset_0' '66496'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_111,subset_185,subset_169,subset_71,subset_47' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_33,subset_42,subset_70' &
distillation 'subset_75,subset_193,subset_56,subset_23,subset_173' 'subset_91,subset_125,subset_49' '66535'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_55,subset_21,subset_120,subset_4,subset_104' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_15,subset_112' &
distillation 'subset_111,subset_185,subset_169,subset_71,subset_47' 'subset_33,subset_42,subset_70' '72633'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_31,subset_138,subset_60,subset_211,subset_16' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_56,subset_41,subset_50' &
distillation 'subset_55,subset_21,subset_120,subset_4,subset_104' 'subset_109,subset_15,subset_112' '2665'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_149,subset_78,subset_103,subset_15,subset_116' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_26,subset_43,subset_79' &
distillation 'subset_31,subset_138,subset_60,subset_211,subset_16' 'subset_56,subset_41,subset_50' '51177'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_106,subset_105,subset_48,subset_2,subset_44' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_5,subset_102,subset_36' &
distillation 'subset_149,subset_78,subset_103,subset_15,subset_116' 'subset_26,subset_43,subset_79' '61621'
distillation 'subset_106,subset_105,subset_48,subset_2,subset_44' 'subset_5,subset_102,subset_36' '61621'
# Epoch 6
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_125,subset_103,subset_41,subset_204,subset_168' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_51,subset_48,subset_38' &
distillation 'subset_106,subset_105,subset_48,subset_2,subset_44' 'subset_5,subset_102,subset_36' '14089'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_135,subset_29,subset_177,subset_179,subset_105' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_93,subset_82,subset_15' &
distillation 'subset_125,subset_103,subset_41,subset_204,subset_168' 'subset_51,subset_48,subset_38' '56482'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_61,subset_210,subset_26,subset_66,subset_6' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_16,subset_91,subset_76' &
distillation 'subset_135,subset_29,subset_177,subset_179,subset_105' 'subset_93,subset_82,subset_15' '17555'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_222,subset_129,subset_158,subset_124,subset_33' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_119,subset_125,subset_23' &
distillation 'subset_61,subset_210,subset_26,subset_66,subset_6' 'subset_16,subset_91,subset_76' '34664'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_180,subset_163,subset_188,subset_56,subset_212' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_56,subset_6,subset_126' &
distillation 'subset_222,subset_129,subset_158,subset_124,subset_33' 'subset_119,subset_125,subset_23' '95355'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_199,subset_140,subset_151,subset_23,subset_27' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_45,subset_85,subset_34' &
distillation 'subset_180,subset_163,subset_188,subset_56,subset_212' 'subset_56,subset_6,subset_126' '47943'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_145,subset_201,subset_3,subset_97,subset_40' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_33,subset_21,subset_105' &
distillation 'subset_199,subset_140,subset_151,subset_23,subset_27' 'subset_45,subset_85,subset_34' '52855'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_85,subset_132,subset_139,subset_182,subset_155' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_77,subset_66,subset_114' &
distillation 'subset_145,subset_201,subset_3,subset_97,subset_40' 'subset_33,subset_21,subset_105' '47964'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_98,subset_169,subset_214,subset_42,subset_106' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_107,subset_49,subset_71' &
distillation 'subset_85,subset_132,subset_139,subset_182,subset_155' 'subset_77,subset_66,subset_114' '5930'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_185,subset_54,subset_22,subset_34,subset_75' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_78,subset_97,subset_81' &
distillation 'subset_98,subset_169,subset_214,subset_42,subset_106' 'subset_107,subset_49,subset_71' '52505'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_69,subset_8,subset_152,subset_217,subset_31' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_12,subset_112,subset_133' &
distillation 'subset_185,subset_54,subset_22,subset_34,subset_75' 'subset_78,subset_97,subset_81' '6705'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_50,subset_28,subset_81,subset_108,subset_189' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_87,subset_18,subset_55' &
distillation 'subset_69,subset_8,subset_152,subset_217,subset_31' 'subset_12,subset_112,subset_133' '74747'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_59,subset_127,subset_21,subset_187,subset_178' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_88,subset_59,subset_68' &
distillation 'subset_50,subset_28,subset_81,subset_108,subset_189' 'subset_87,subset_18,subset_55' '73638'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_122,subset_84,subset_43,subset_123,subset_109' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_80,subset_47,subset_74' &
distillation 'subset_59,subset_127,subset_21,subset_187,subset_178' 'subset_88,subset_59,subset_68' '25513'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_136,subset_86,subset_156,subset_96,subset_58' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_1,subset_79,subset_4' &
distillation 'subset_122,subset_84,subset_43,subset_123,subset_109' 'subset_80,subset_47,subset_74' '47516'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_89,subset_116,subset_0,subset_146,subset_20' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_44,subset_32,subset_57' &
distillation 'subset_136,subset_86,subset_156,subset_96,subset_58' 'subset_1,subset_79,subset_4' '72512'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_142,subset_74,subset_157,subset_166,subset_206' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_43,subset_36,subset_20' &
distillation 'subset_89,subset_116,subset_0,subset_146,subset_20' 'subset_44,subset_32,subset_57' '37827'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_118,subset_153,subset_112,subset_197,subset_7' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_35,subset_17,subset_102' &
distillation 'subset_142,subset_74,subset_157,subset_166,subset_206' 'subset_43,subset_36,subset_20' '9637'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_147,subset_62,subset_207,subset_11,subset_36' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_83,subset_61' &
distillation 'subset_118,subset_153,subset_112,subset_197,subset_7' 'subset_35,subset_17,subset_102' '50639'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_67,subset_19,subset_80,subset_32,subset_161' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_103,subset_106' &
distillation 'subset_147,subset_62,subset_207,subset_11,subset_36' 'subset_3,subset_83,subset_61' '66093'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_154,subset_10,subset_134,subset_172,subset_160' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_0,subset_28,subset_110' &
distillation 'subset_67,subset_19,subset_80,subset_32,subset_161' 'subset_129,subset_103,subset_106' '59028'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_110,subset_159,subset_76,subset_137,subset_167' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_100,subset_94,subset_131' &
distillation 'subset_154,subset_10,subset_134,subset_172,subset_160' 'subset_0,subset_28,subset_110' '72070'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_37,subset_130,subset_192,subset_186,subset_208' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_122,subset_117,subset_72' &
distillation 'subset_110,subset_159,subset_76,subset_137,subset_167' 'subset_100,subset_94,subset_131' '36663'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_176,subset_79,subset_65,subset_92,subset_165' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_75,subset_29,subset_84' &
distillation 'subset_37,subset_130,subset_192,subset_186,subset_208' 'subset_122,subset_117,subset_72' '81689'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_200,subset_114,subset_198,subset_55,subset_17' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_67,subset_22,subset_116' &
distillation 'subset_176,subset_79,subset_65,subset_92,subset_165' 'subset_75,subset_29,subset_84' '89181'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_53,subset_193,subset_13,subset_133,subset_149' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_127,subset_89,subset_41' &
distillation 'subset_200,subset_114,subset_198,subset_55,subset_17' 'subset_67,subset_22,subset_116' '80090'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_44,subset_47,subset_221,subset_39,subset_220' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_37,subset_123,subset_50' &
distillation 'subset_53,subset_193,subset_13,subset_133,subset_149' 'subset_127,subset_89,subset_41' '15572'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_143,subset_64,subset_63,subset_117,subset_68' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_135,subset_113,subset_5' &
distillation 'subset_44,subset_47,subset_221,subset_39,subset_220' 'subset_37,subset_123,subset_50' '16864'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_46,subset_190,subset_196,subset_72,subset_138' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_111,subset_96,subset_62' &
distillation 'subset_143,subset_64,subset_63,subset_117,subset_68' 'subset_135,subset_113,subset_5' '12672'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_173,subset_16,subset_99,subset_144,subset_94' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_124,subset_64,subset_120' &
distillation 'subset_46,subset_190,subset_196,subset_72,subset_138' 'subset_111,subset_96,subset_62' '51629'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_183,subset_30,subset_25,subset_51,subset_175' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_136,subset_70,subset_63' &
distillation 'subset_173,subset_16,subset_99,subset_144,subset_94' 'subset_124,subset_64,subset_120' '48915'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_219,subset_12,subset_18,subset_70,subset_38' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_13,subset_130,subset_86' &
distillation 'subset_183,subset_30,subset_25,subset_51,subset_175' 'subset_136,subset_70,subset_63' '44444'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_216,subset_5,subset_52,subset_119,subset_93' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_14,subset_46,subset_24' &
distillation 'subset_219,subset_12,subset_18,subset_70,subset_38' 'subset_13,subset_130,subset_86' '73116'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_102,subset_1,subset_113,subset_205,subset_9' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_9,subset_132,subset_39' &
distillation 'subset_216,subset_5,subset_52,subset_119,subset_93' 'subset_14,subset_46,subset_24' '47920'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_174,subset_104,subset_191,subset_57,subset_2' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_42,subset_31,subset_10' &
distillation 'subset_102,subset_1,subset_113,subset_205,subset_9' 'subset_9,subset_132,subset_39' '98954'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_91,subset_35,subset_171,subset_213,subset_126' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_60,subset_2,subset_54' &
distillation 'subset_174,subset_104,subset_191,subset_57,subset_2' 'subset_42,subset_31,subset_10' '18909'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_120,subset_194,subset_195,subset_184,subset_203' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_99,subset_104,subset_95' &
distillation 'subset_91,subset_35,subset_171,subset_213,subset_126' 'subset_60,subset_2,subset_54' '26090'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_128,subset_4,subset_131,subset_45,subset_141' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_52,subset_11,subset_90' &
distillation 'subset_120,subset_194,subset_195,subset_184,subset_203' 'subset_99,subset_104,subset_95' '78966'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_121,subset_24,subset_71,subset_88,subset_111' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_115,subset_53,subset_30' &
distillation 'subset_128,subset_4,subset_131,subset_45,subset_141' 'subset_52,subset_11,subset_90' '66731'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_215,subset_49,subset_218,subset_87,subset_82' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_92,subset_58,subset_19' &
distillation 'subset_121,subset_24,subset_71,subset_88,subset_111' 'subset_115,subset_53,subset_30' '52610'
distillation 'subset_215,subset_49,subset_218,subset_87,subset_82' 'subset_92,subset_58,subset_19' '52610'
# Epoch 7
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_49,subset_103,subset_117,subset_212,subset_157' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_33,subset_121,subset_47' &
distillation 'subset_215,subset_49,subset_218,subset_87,subset_82' 'subset_92,subset_58,subset_19' '455'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_189,subset_177,subset_68,subset_172,subset_58' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_39,subset_120,subset_92' &
distillation 'subset_49,subset_103,subset_117,subset_212,subset_157' 'subset_33,subset_121,subset_47' '99113'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_90,subset_209,subset_205,subset_111,subset_37' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_84,subset_26,subset_114' &
distillation 'subset_189,subset_177,subset_68,subset_172,subset_58' 'subset_39,subset_120,subset_92' '2979'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_155,subset_188,subset_56,subset_178,subset_67' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_90,subset_3,subset_8' &
distillation 'subset_90,subset_209,subset_205,subset_111,subset_37' 'subset_84,subset_26,subset_114' '39107'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_31,subset_97,subset_1,subset_215,subset_7' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_123,subset_36,subset_16' &
distillation 'subset_155,subset_188,subset_56,subset_178,subset_67' 'subset_90,subset_3,subset_8' '60686'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_77,subset_137,subset_129,subset_174,subset_13' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_137,subset_118,subset_56' &
distillation 'subset_31,subset_97,subset_1,subset_215,subset_7' 'subset_123,subset_36,subset_16' '88276'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_71,subset_161,subset_163,subset_175,subset_33' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_23,subset_134,subset_98' &
distillation 'subset_77,subset_137,subset_129,subset_174,subset_13' 'subset_137,subset_118,subset_56' '94253'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_84,subset_8,subset_11,subset_136,subset_181' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_54,subset_20,subset_132' &
distillation 'subset_71,subset_161,subset_163,subset_175,subset_33' 'subset_23,subset_134,subset_98' '71361'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_165,subset_185,subset_45,subset_193,subset_107' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_104,subset_109,subset_131' &
distillation 'subset_84,subset_8,subset_11,subset_136,subset_181' 'subset_54,subset_20,subset_132' '55591'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_41,subset_54,subset_22,subset_131,subset_6' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_9,subset_111,subset_83' &
distillation 'subset_165,subset_185,subset_45,subset_193,subset_107' 'subset_104,subset_109,subset_131' '69721'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_93,subset_50,subset_48,subset_164,subset_204' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_4,subset_127,subset_60' &
distillation 'subset_41,subset_54,subset_22,subset_131,subset_6' 'subset_9,subset_111,subset_83' '49712'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_92,subset_4,subset_122,subset_15,subset_160' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_66,subset_108,subset_38' &
distillation 'subset_93,subset_50,subset_48,subset_164,subset_204' 'subset_4,subset_127,subset_60' '30102'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_143,subset_211,subset_20,subset_183,subset_73' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_82,subset_70,subset_51' &
distillation 'subset_92,subset_4,subset_122,subset_15,subset_160' 'subset_66,subset_108,subset_38' '32441'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_180,subset_125,subset_130,subset_156,subset_59' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_85,subset_30,subset_58' &
distillation 'subset_143,subset_211,subset_20,subset_183,subset_73' 'subset_82,subset_70,subset_51' '60375'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_214,subset_82,subset_26,subset_0,subset_27' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_50,subset_110,subset_43' &
distillation 'subset_180,subset_125,subset_130,subset_156,subset_59' 'subset_85,subset_30,subset_58' '45371'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_38,subset_39,subset_126,subset_201,subset_145' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_25,subset_62,subset_88' &
distillation 'subset_214,subset_82,subset_26,subset_0,subset_27' 'subset_50,subset_110,subset_43' '20332'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_89,subset_78,subset_104,subset_158,subset_203' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_17,subset_71,subset_75' &
distillation 'subset_38,subset_39,subset_126,subset_201,subset_145' 'subset_25,subset_62,subset_88' '36129'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_186,subset_40,subset_123,subset_199,subset_151' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_76,subset_81,subset_93' &
distillation 'subset_89,subset_78,subset_104,subset_158,subset_203' 'subset_17,subset_71,subset_75' '24704'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_176,subset_3,subset_196,subset_28,subset_134' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_2,subset_13,subset_133' &
distillation 'subset_186,subset_40,subset_123,subset_199,subset_151' 'subset_76,subset_81,subset_93' '95030'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_170,subset_62,subset_167,subset_135,subset_119' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_15,subset_42,subset_46' &
distillation 'subset_176,subset_3,subset_196,subset_28,subset_134' 'subset_2,subset_13,subset_133' '14790'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_124,subset_173,subset_208,subset_69,subset_102' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_44,subset_6,subset_79' &
distillation 'subset_170,subset_62,subset_167,subset_135,subset_119' 'subset_15,subset_42,subset_46' '4222'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_109,subset_87,subset_96,subset_57,subset_190' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_19,subset_61,subset_74' &
distillation 'subset_124,subset_173,subset_208,subset_69,subset_102' 'subset_44,subset_6,subset_79' '86533'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_216,subset_207,subset_10,subset_218,subset_63' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_77,subset_31,subset_49' &
distillation 'subset_109,subset_87,subset_96,subset_57,subset_190' 'subset_19,subset_61,subset_74' '54950'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_127,subset_98,subset_52,subset_171,subset_197' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_10,subset_52,subset_11' &
distillation 'subset_216,subset_207,subset_10,subset_218,subset_63' 'subset_77,subset_31,subset_49' '80599'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_140,subset_46,subset_132,subset_32,subset_51' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_126,subset_67,subset_119' &
distillation 'subset_127,subset_98,subset_52,subset_171,subset_197' 'subset_10,subset_52,subset_11' '2050'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_17,subset_2,subset_81,subset_24,subset_86' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_53,subset_86,subset_129' &
distillation 'subset_140,subset_46,subset_132,subset_32,subset_51' 'subset_126,subset_67,subset_119' '31544'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_187,subset_75,subset_55,subset_88,subset_147' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_117,subset_125,subset_89' &
distillation 'subset_17,subset_2,subset_81,subset_24,subset_86' 'subset_53,subset_86,subset_129' '27011'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_112,subset_65,subset_106,subset_60,subset_19' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_112,subset_78,subset_72' &
distillation 'subset_187,subset_75,subset_55,subset_88,subset_147' 'subset_117,subset_125,subset_89' '8819'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_91,subset_9,subset_162,subset_72,subset_217' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_40,subset_29,subset_102' &
distillation 'subset_112,subset_65,subset_106,subset_60,subset_19' 'subset_112,subset_78,subset_72' '13224'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_115,subset_150,subset_222,subset_200,subset_83' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_128,subset_41,subset_103' &
distillation 'subset_91,subset_9,subset_162,subset_72,subset_217' 'subset_40,subset_29,subset_102' '77849'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_213,subset_152,subset_110,subset_95,subset_76' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_21,subset_28,subset_34' &
distillation 'subset_115,subset_150,subset_222,subset_200,subset_83' 'subset_128,subset_41,subset_103' '4400'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_142,subset_192,subset_202,subset_34,subset_128' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_130,subset_97,subset_37' &
distillation 'subset_213,subset_152,subset_110,subset_95,subset_76' 'subset_21,subset_28,subset_34' '58515'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_16,subset_179,subset_21,subset_139,subset_80' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_73,subset_59,subset_135' &
distillation 'subset_142,subset_192,subset_202,subset_34,subset_128' 'subset_130,subset_97,subset_37' '78352'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_70,subset_120,subset_220,subset_42,subset_198' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_95,subset_65,subset_99' &
distillation 'subset_16,subset_179,subset_21,subset_139,subset_80' 'subset_73,subset_59,subset_135' '88083'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_184,subset_66,subset_108,subset_99,subset_144' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_27,subset_136,subset_94' &
distillation 'subset_70,subset_120,subset_220,subset_42,subset_198' 'subset_95,subset_65,subset_99' '92181'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_113,subset_43,subset_121,subset_25,subset_29' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_124,subset_35,subset_105' &
distillation 'subset_184,subset_66,subset_108,subset_99,subset_144' 'subset_27,subset_136,subset_94' '6380'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_5,subset_35,subset_146,subset_94,subset_159' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_80,subset_106,subset_45' &
distillation 'subset_113,subset_43,subset_121,subset_25,subset_29' 'subset_124,subset_35,subset_105' '32077'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_23,subset_191,subset_206,subset_36,subset_168' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_96,subset_91,subset_12' &
distillation 'subset_5,subset_35,subset_146,subset_94,subset_159' 'subset_80,subset_106,subset_45' '97064'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_101,subset_14,subset_138,subset_194,subset_53' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_64,subset_32,subset_5' &
distillation 'subset_23,subset_191,subset_206,subset_36,subset_168' 'subset_96,subset_91,subset_12' '5798'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_64,subset_44,subset_149,subset_153,subset_30' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_87,subset_101,subset_22' &
distillation 'subset_101,subset_14,subset_138,subset_194,subset_53' 'subset_64,subset_32,subset_5' '52725'
distillation 'subset_64,subset_44,subset_149,subset_153,subset_30' 'subset_87,subset_101,subset_22' '52725'
# Epoch 8
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_176,subset_93,subset_74,subset_168,subset_21' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_32,subset_26,subset_82' &
distillation 'subset_64,subset_44,subset_149,subset_153,subset_30' 'subset_87,subset_101,subset_22' '35180'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_85,subset_128,subset_129,subset_123,subset_40' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_29,subset_46,subset_6' &
distillation 'subset_176,subset_93,subset_74,subset_168,subset_21' 'subset_32,subset_26,subset_82' '64634'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_7,subset_144,subset_32,subset_191,subset_14' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_77,subset_2,subset_99' &
distillation 'subset_85,subset_128,subset_129,subset_123,subset_40' 'subset_29,subset_46,subset_6' '19127'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_209,subset_216,subset_179,subset_89,subset_125' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_127,subset_121,subset_81' &
distillation 'subset_7,subset_144,subset_32,subset_191,subset_14' 'subset_77,subset_2,subset_99' '8292'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_19,subset_6,subset_35,subset_57,subset_187' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_60,subset_12,subset_24' &
distillation 'subset_209,subset_216,subset_179,subset_89,subset_125' 'subset_127,subset_121,subset_81' '21916'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_50,subset_82,subset_1,subset_30,subset_188' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_129,subset_112,subset_11' &
distillation 'subset_19,subset_6,subset_35,subset_57,subset_187' 'subset_60,subset_12,subset_24' '56973'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_146,subset_170,subset_145,subset_28,subset_195' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_109,subset_22,subset_103' &
distillation 'subset_50,subset_82,subset_1,subset_30,subset_188' 'subset_129,subset_112,subset_11' '36170'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_201,subset_101,subset_100,subset_202,subset_124' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_63,subset_52,subset_35' &
distillation 'subset_146,subset_170,subset_145,subset_28,subset_195' 'subset_109,subset_22,subset_103' '55194'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_103,subset_121,subset_17,subset_218,subset_65' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_118,subset_135,subset_66' &
distillation 'subset_201,subset_101,subset_100,subset_202,subset_124' 'subset_63,subset_52,subset_35' '39289'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_220,subset_213,subset_25,subset_130,subset_110' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_100,subset_36,subset_34' &
distillation 'subset_103,subset_121,subset_17,subset_218,subset_65' 'subset_118,subset_135,subset_66' '63469'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_34,subset_115,subset_42,subset_73,subset_154' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_95,subset_114,subset_19' &
distillation 'subset_220,subset_213,subset_25,subset_130,subset_110' 'subset_100,subset_36,subset_34' '10082'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_133,subset_139,subset_81,subset_151,subset_172' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_89,subset_125,subset_111' &
distillation 'subset_34,subset_115,subset_42,subset_73,subset_154' 'subset_95,subset_114,subset_19' '47333'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_69,subset_49,subset_159,subset_177,subset_58' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_134,subset_33,subset_133' &
distillation 'subset_133,subset_139,subset_81,subset_151,subset_172' 'subset_89,subset_125,subset_111' '32990'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_157,subset_68,subset_200,subset_38,subset_59' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_117,subset_107,subset_37' &
distillation 'subset_69,subset_49,subset_159,subset_177,subset_58' 'subset_134,subset_33,subset_133' '32321'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_56,subset_0,subset_98,subset_97,subset_15' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_62,subset_88,subset_18' &
distillation 'subset_157,subset_68,subset_200,subset_38,subset_59' 'subset_117,subset_107,subset_37' '94489'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_120,subset_86,subset_163,subset_193,subset_96' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_93,subset_42,subset_119' &
distillation 'subset_56,subset_0,subset_98,subset_97,subset_15' 'subset_62,subset_88,subset_18' '81924'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_207,subset_161,subset_135,subset_140,subset_13' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_14,subset_49,subset_27' &
distillation 'subset_120,subset_86,subset_163,subset_193,subset_96' 'subset_93,subset_42,subset_119' '64985'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_45,subset_77,subset_67,subset_11,subset_166' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_74,subset_137,subset_76' &
distillation 'subset_207,subset_161,subset_135,subset_140,subset_13' 'subset_14,subset_49,subset_27' '77831'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_3,subset_221,subset_51,subset_63,subset_117' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_69,subset_85,subset_72' &
distillation 'subset_45,subset_77,subset_67,subset_11,subset_166' 'subset_74,subset_137,subset_76' '80828'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_9,subset_83,subset_174,subset_26,subset_189' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_68,subset_104,subset_57' &
distillation 'subset_3,subset_221,subset_51,subset_63,subset_117' 'subset_69,subset_85,subset_72' '25630'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_114,subset_113,subset_5,subset_137,subset_165' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_70,subset_67,subset_8' &
distillation 'subset_9,subset_83,subset_174,subset_26,subset_189' 'subset_68,subset_104,subset_57' '60012'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_29,subset_18,subset_196,subset_208,subset_94' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_86,subset_17,subset_25' &
distillation 'subset_114,subset_113,subset_5,subset_137,subset_165' 'subset_70,subset_67,subset_8' '14021'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_197,subset_183,subset_76,subset_54,subset_175' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_55,subset_79,subset_15' &
distillation 'subset_29,subset_18,subset_196,subset_208,subset_94' 'subset_86,subset_17,subset_25' '17758'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_181,subset_41,subset_158,subset_217,subset_79' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_45,subset_39,subset_58' &
distillation 'subset_197,subset_183,subset_76,subset_54,subset_175' 'subset_55,subset_79,subset_15' '39932'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_55,subset_70,subset_192,subset_185,subset_53' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_97,subset_122,subset_130' &
distillation 'subset_181,subset_41,subset_158,subset_217,subset_79' 'subset_45,subset_39,subset_58' '887'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_75,subset_52,subset_173,subset_12,subset_142' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_5,subset_13,subset_113' &
distillation 'subset_55,subset_70,subset_192,subset_185,subset_53' 'subset_97,subset_122,subset_130' '51864'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_210,subset_71,subset_91,subset_204,subset_212' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_84,subset_43,subset_61' &
distillation 'subset_75,subset_52,subset_173,subset_12,subset_142' 'subset_5,subset_13,subset_113' '43544'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_141,subset_186,subset_164,subset_132,subset_105' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_3,subset_56,subset_105' &
distillation 'subset_210,subset_71,subset_91,subset_204,subset_212' 'subset_84,subset_43,subset_61' '81289'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_150,subset_178,subset_90,subset_169,subset_156' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_7,subset_124,subset_102' &
distillation 'subset_141,subset_186,subset_164,subset_132,subset_105' 'subset_3,subset_56,subset_105' '49792'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_111,subset_131,subset_184,subset_215,subset_134' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_71,subset_21,subset_4' &
distillation 'subset_150,subset_178,subset_90,subset_169,subset_156' 'subset_7,subset_124,subset_102' '43193'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_127,subset_31,subset_203,subset_48,subset_180' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_94,subset_98,subset_51' &
distillation 'subset_111,subset_131,subset_184,subset_215,subset_134' 'subset_71,subset_21,subset_4' '57700'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_20,subset_119,subset_33,subset_112,subset_23' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_80,subset_75,subset_28' &
distillation 'subset_127,subset_31,subset_203,subset_48,subset_180' 'subset_94,subset_98,subset_51' '43874'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_43,subset_44,subset_72,subset_118,subset_171' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_23,subset_53,subset_131' &
distillation 'subset_20,subset_119,subset_33,subset_112,subset_23' 'subset_80,subset_75,subset_28' '56441'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_116,subset_219,subset_122,subset_211,subset_46' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_48,subset_30,subset_20' &
distillation 'subset_43,subset_44,subset_72,subset_118,subset_171' 'subset_23,subset_53,subset_131' '85332'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_106,subset_22,subset_62,subset_64,subset_136' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_120,subset_40,subset_91' &
distillation 'subset_116,subset_219,subset_122,subset_211,subset_46' 'subset_48,subset_30,subset_20' '77908'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_102,subset_87,subset_182,subset_147,subset_167' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_10,subset_38,subset_128' &
distillation 'subset_106,subset_22,subset_62,subset_64,subset_136' 'subset_120,subset_40,subset_91' '17842'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_66,subset_80,subset_199,subset_205,subset_2' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_90,subset_101,subset_108' &
distillation 'subset_102,subset_87,subset_182,subset_147,subset_167' 'subset_10,subset_38,subset_128' '39378'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_138,subset_190,subset_99,subset_16,subset_107' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_44,subset_1,subset_41' &
distillation 'subset_66,subset_80,subset_199,subset_205,subset_2' 'subset_90,subset_101,subset_108' '42030'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_36,subset_152,subset_162,subset_155,subset_214' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_0,subset_9,subset_83' &
distillation 'subset_138,subset_190,subset_99,subset_16,subset_107' 'subset_44,subset_1,subset_41' '79064'
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.reazon_speech_all.wer_10.0.vectorized' -c 'subset_39,subset_109,subset_4,subset_88,subset_198' &
python ./misc/hf_dataset_download.py -d 'japanese-asr/whisper_transcriptions.mls.wer_10.0.vectorized' -c 'subset_92,subset_87,subset_31' &
distillation 'subset_36,subset_152,subset_162,subset_155,subset_214' 'subset_0,subset_9,subset_83' '91133'
distillation 'subset_39,subset_109,subset_4,subset_88,subset_198' 'subset_92,subset_87,subset_31' '91133'
