"""Initialize a student Whisper model from a pre-trained teacher model for teacher-student distillation."""
import argparse
import copy
import logging
import os

import numpy as np
import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ['CURL_CA_BUNDLE'] = ''

# disable warning message
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialise a student Whisper model from a teacher model, copying the relevant layer weights and adjusting the processor as necessary."
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        required=True,
        help="The HF Hub ID of the teacher checkpoint.",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=None,
        help="Number of encoder layers to use in the student model. Defaults to all layers from the teacher.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=2,
        help="Number of decoder layers to use in the student model. Defaults to 2 layers.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Where to save the student weights and processor.",
    )
    args = parser.parse_args()
    return args


def init_student_model_from_teacher(
    teacher_checkpoint,
    save_dir,
    encoder_layers=None,
    decoder_layers=2,
):
    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        teacher_checkpoint,
        low_cpu_mem_usage=True,
    )
    processor = WhisperProcessor.from_pretrained(teacher_checkpoint)
    generation_config = GenerationConfig.from_pretrained(teacher_checkpoint)

    teacher_config = teacher_model.config
    teacher_encoder_layers = teacher_config.encoder_layers
    teacher_decoder_layers = teacher_config.decoder_layers

    student_config = copy.deepcopy(teacher_config)
    student_config.update(
        {
            "encoder_layers": encoder_layers if encoder_layers is not None else teacher_encoder_layers,
            "decoder_layers": decoder_layers,
        }
    )

    encoder_mapping = np.linspace(0, teacher_encoder_layers - 1, student_config.encoder_layers, dtype=int)
    encoder_mapping[-1] = teacher_encoder_layers - 1

    encoder_map = {}
    for student_layer, teacher_layer in enumerate(encoder_mapping):
        encoder_map[teacher_layer] = student_layer

    decoder_mapping = np.linspace(0, teacher_decoder_layers - 1, student_config.decoder_layers, dtype=int)
    decoder_mapping[-1] = teacher_decoder_layers - 1

    decoder_map = {}
    for student_layer, teacher_layer in enumerate(decoder_mapping):
        decoder_map[teacher_layer] = student_layer

    # init the student params from the teacher model
    student_model = WhisperForConditionalGeneration(student_config)
    missing_keys, unexpected_keys = student_model.load_state_dict(teacher_model.state_dict(), strict=False)
    if len(missing_keys) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for WhisperForConditionalGeneration. \n"
            f"Missing key(s) in state_dict: {missing_keys}"
        )
    if decoder_layers == teacher_decoder_layers:
        decoder_keys = [key for key in unexpected_keys if "model.decoder.layers" in key]
        if len(decoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for WhisperForConditionalGeneration. \n"
                f"Unexpected key(s) in state_dict: {decoder_keys}"
            )
    if encoder_layers == teacher_encoder_layers:
        encoder_keys = [key for key in unexpected_keys if "model.encoder.layers" in key]
        if len(encoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for WhisperForConditionalGeneration. \n"
                f"Unexpected key(s) in state_dict: {encoder_keys}"
            )

    for layer in range(teacher_decoder_layers):
        if layer in decoder_map:
            # re-introduce pre-defined layers from the teacher
            student_model.model.decoder.layers[decoder_map[layer]].load_state_dict(
                teacher_model.model.decoder.layers[layer].state_dict()
            )

    if encoder_layers is not None:
        for layer in range(teacher_encoder_layers):
            if layer in encoder_map:
                # re-introduce pre-defined layers from the teacher
                student_model.model.encoder.layers[encoder_map[layer]].load_state_dict(
                    teacher_model.model.encoder.layers[layer].state_dict()
                )

    # remove the teacher params and model
    del teacher_model

    # save the converted weights and model
    student_model.save_pretrained(save_dir)
    # we also need to correctly save the processor and generation config
    processor.save_pretrained(save_dir)
    generation_config.save_pretrained(save_dir)

    # check we can do a forward pass with the saved model - first load the weights and processor
    logger.info("Checking we can load the saved model...")
    student_model = WhisperForConditionalGeneration.from_pretrained(save_dir, low_cpu_mem_usage=True)
    processor = WhisperProcessor.from_pretrained(save_dir)

    # define some random inputs
    input_features = processor(np.ones(16000), sampling_rate=16000, return_tensors="pt").input_features
    decoder_start_token_id = student_model.config.decoder_start_token_id
    decoder_input_ids = torch.ones((input_features.shape[0], 1), dtype=torch.long) * decoder_start_token_id

    # do a forward pass - outputs will be gibberish for the initialised model so we can't check them
    # but we make can sure the model runs as expected
    logger.info("Checking we can run the converted model forward...")
    _ = student_model(input_features, decoder_input_ids=decoder_input_ids).logits
    logger.info("Conversion successful!")


if __name__ == "__main__":
    args = parse_args()

    init_student_model_from_teacher(
        teacher_checkpoint=args.teacher_checkpoint,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        save_dir=args.save_dir,
    )
