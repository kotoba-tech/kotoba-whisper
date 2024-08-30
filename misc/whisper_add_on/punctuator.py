"""Add punctuation."""
from typing import List, Dict, Any
from punctuators.models import PunctCapSegModelONNX


class Punctuator:

    ja_punctuations = ["!", "?", "、", "。"]

    def __init__(self, model: str = "pcs_47lang"):
        self.punctuation_model = PunctCapSegModelONNX.from_pretrained(model)

    def punctuate(self, pipeline_chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        def validate_punctuation(raw: str, punctuated: str):
            if 'unk' in punctuated:
                return raw
            if punctuated.count("。") > 1:
                ind = punctuated.rfind("。")
                punctuated = punctuated.replace("。", "")
                punctuated = punctuated[:ind] + "。" + punctuated[ind:]
            return punctuated

        text_edit = self.punctuation_model.infer([c['text'] for c in pipeline_chunk])
        return [
            {
                'timestamp': c['timestamp'],
                'text': validate_punctuation(c['text'], "".join(e))
            } for c, e in zip(pipeline_chunk, text_edit)
        ]


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
    punc = Punctuator()
    for i in dataset:
        # transcribe the audio
        audio = i["audio"]
        if audio["path"] == "long_interview_1.mp3":
            audio["array"] = audio["array"][:7938000]
        generate_kwargs = {"language": "japanese", "task": "transcribe"}
        prediction = pipe(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
        # fix the timestamp
        prediction_punctuated = punc.punctuate(prediction['chunks'])
        # compare the transcriptions
        pprint(f'FILE:{i["audio"]["path"]}')
        for a, b in zip(prediction["chunks"], prediction_punctuated):
            pprint("*RAW*")
            pprint(a)
            pprint("*PUNCTUATED*")
            pprint(b)
            print()
        input()
