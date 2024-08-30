"""Adapt code from https://github.com/jianfch/stable-ts/blob/main/stable_whisper/whisper_word_level/hf_whisper.py
pip install stable-ts==2.16.0
"""
from typing import List, Dict, Any

import numpy as np
from stable_whisper import WhisperResult


def _fix_timestamp(sample_rate: int, result: List[Dict[str, Any]], audio: np.ndarray) -> WhisperResult or None:

    def replace_none_ts(parts):
        total_dur = round(audio.shape[-1] / sample_rate, 3)
        _medium_dur = _ts_nonzero_mask = None

        def ts_nonzero_mask() -> np.ndarray:
            nonlocal _ts_nonzero_mask
            if _ts_nonzero_mask is None:
                _ts_nonzero_mask = np.array([(p['end'] or p['start']) is not None for p in parts])
            return _ts_nonzero_mask

        def medium_dur() -> float:
            nonlocal _medium_dur
            if _medium_dur is None:
                nonzero_dus = [p['end'] - p['start'] for p in parts if None not in (p['end'], p['start'])]
                nonzero_durs = np.array(nonzero_dus)
                _medium_dur = np.median(nonzero_durs) * 2 if len(nonzero_durs) else 2.0
            return _medium_dur

        def _curr_max_end(start: float, next_idx: float) -> float:
            max_end = total_dur
            if next_idx != len(parts):
                mask = np.flatnonzero(ts_nonzero_mask()[next_idx:])
                if len(mask):
                    _part = parts[mask[0]+next_idx]
                    max_end = _part['start'] or _part['end']

            new_end = round(start + medium_dur(), 3)
            if new_end > max_end:
                return max_end
            return new_end

        for i, part in enumerate(parts, 1):
            if part['start'] is None:
                is_first = i == 1
                if is_first:
                    new_start = round((part['end'] or 0) - medium_dur(), 3)
                    part['start'] = max(new_start, 0.0)
                else:
                    part['start'] = parts[i - 2]['end']
            if part['end'] is None:
                no_next_start = i == len(parts) or parts[i]['start'] is None
                part['end'] = _curr_max_end(part['start'], i) if no_next_start else parts[i]['start']

    words = [dict(start=word['timestamp'][0], end=word['timestamp'][1], word=word['text']) for word in result]
    replace_none_ts(words)
    return WhisperResult([words], force_order=True, check_sorted=True)


def fix_timestamp(pipeline_output: List[Dict[str, Any]], audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
    result = _fix_timestamp(sample_rate=sample_rate, audio=audio, result=pipeline_output)
    result.adjust_by_silence(
        audio,
        q_levels=20,
        k_size=5,
        sample_rate=sample_rate,
        min_word_dur=None,
        word_level=True,
        verbose=True,
        nonspeech_error=0.1,
        use_word_position=True
    )
    if result.has_words:
        result.regroup(True)
    return [{"timestamp": [s.start, s.end], "text": s.text} for s in result.segments]


if __name__ == '__main__':
    from pprint import pprint
    from datasets import load_dataset
    from transformers import pipeline

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v1.0",
        chunk_length_s=15,
        batch_size=4,
    )
    dataset = load_dataset("kotoba-tech/kotoba-whisper-eval", split="train")
    for i in dataset:
        # transcribe the audio
        audio = i["audio"]
        if audio["path"] == "long_interview_1.mp3":
            audio["array"] = audio["array"][:7938000]
        generate_kwargs = {"language": "japanese", "task": "transcribe"}
        prediction = pipe(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
        # fix the timestamp
        prediction_fixed = fix_timestamp(
            pipeline_output=prediction["chunks"],
            audio=i["audio"]["array"],
            sample_rate=i["audio"]["sampling_rate"]
        )
        # compare the transcriptions
        pprint(f'FILE:{i["audio"]["path"]}')
        for a, b in zip(prediction["chunks"], prediction_fixed):
            pprint("*RAW*")
            pprint(a)
            pprint("*FIXED*")
            pprint(b)
            print()
        input()

