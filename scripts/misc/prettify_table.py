import pandas as pd

from glob import glob
import evaluate
from yakinori import Yakinori


wer = evaluate.load("wer")
cer = evaluate.load("cer")
yakinori = Yakinori()


def text_to_pronounce(text: str):
    parsed_list = yakinori.get_parsed_list(text)
    return yakinori.get_roma_sentence(parsed_list, is_hatsuon=True)


def normalization(text: str):
    text = text.replace(" ", "")
    return text


if __name__ == '__main__':
    summary = []
    pred_list = {}
    for i in glob("eval/*/all_predictions.test.csv"):
        model = i.split("/")[1].split(".")[0]
        data = i.split("/")[1].split(".")[-1]
        df = pd.read_csv(i, index_col=0)
        target = [normalization(i) for i in df["Norm Target"].values]
        pred = [normalization(i) for i in df["Norm Pred"].values]
        score_wer = wer.compute(predictions=pred, references=target) * 100
        score_cer = cer.compute(predictions=pred, references=target) * 100
        pred_pr = [text_to_pronounce(p) for p in pred]
        target_pr = [text_to_pronounce(t) for t in target]
        per = cer.compute(predictions=pred_pr, references=target_pr) * 100
        summary.append({"model": model, "data": data, "cer": score_cer, "wer": score_wer, "per": per})
        if data not in pred_list:
            pred_list[data] = {}
        if "target" not in pred_list[data]:
            pred_list[data]["target"] = target
        pred_list[data][model] = pred

    df = pd.DataFrame(summary)
    df = df.sort_values(by=["data", "model"])
    df.to_csv("eval/metric.csv")
    for k, v in pred_list.items():
        pd.DataFrame(v).to_csv(f"eval/prediction.{k}.csv")

    df = pd.read_csv("eval/metric.csv", index_col=0)
    df = df.round(2)
    df["model"] = [
        f"[japanese-asr/{i}](https://huggingface.co/japanese-asr/{i})" if "distil" in i else f"[openai/{i}](https://huggingface.co/openai/{i})"
        for i in df.model
    ]
    # df["data"] = [f"[japanese-asr/{i}](https://huggingface.co/datasets/japanese-asr/{i})" for i in df["data"]]

    print("########### WER ###########")
    print(df.pivot(index="model", columns="data", values="wer").to_markdown())
    print("########### CER ###########")
    print(df.pivot(index="model", columns="data", values="cer").to_markdown())
    print("########### PER ###########")
    print(df.pivot(index="model", columns="data", values="per").to_markdown())
