import transformers
import torch

from train import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, MAX_LENGTH
from utils import names

from collections import defaultdict
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

OUTPUT_DIR = "custom_outputs"
CHECKPOINT = "checkpoint-42455"
NROWS = 2173762
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    # model = transformers.GPT2LMHeadModel.from_pretrained(f"{OUTPUT_DIR}/{CHECKPOINT}")
    # tokenizer = transformers.PreTrainedTokenizerFast(
    #     model_max_length=MAX_LENGTH,
    #     padding_side="right",
    #     tokenizer_file="tokenizers/star2000_tokenizer.json",
    # )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        f"{OUTPUT_DIR}/{CHECKPOINT}"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        f"{OUTPUT_DIR}/{CHECKPOINT}/"
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    tokenizer.add_special_tokens(special_tokens_dict)

    return (model, tokenizer)


def load_lines(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def predict_test():
    prompts = ["0000000$", "0000001$", "0000002$", "0000003$"]
    model, tokenizer = get_model_and_tokenizer()

    inputs = tokenizer(prompts, return_tensors="pt").input_ids

    max_new_tokens = MAX_LENGTH + 10
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    for pred in predictions:
        print(pred)


def predict(batch_size=256):
    prompts = [f"{i:07}$" for i in range(NROWS)]
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)
    max_new_tokens = MAX_LENGTH + 10
    predictions = []
    with open(f"data/{CHECKPOINT}.txt", "w") as f:
        for start_idx in range(0, NROWS, batch_size):
            batch = prompts[start_idx : start_idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            outputs = [
                text.replace(" ", "")
                .replace(DEFAULT_EOS_TOKEN, "")
                .replace(DEFAULT_PAD_TOKEN, "")
                for text in outputs
            ]
            predictions.extend(outputs)
            f.writelines([f"{line}\n" for line in outputs])
    return predictions


def parse_pred_line(line: str, missings, print_missing=False):
    d = {}
    is_missing = False
    for name in names:
        segs = line.split(f"{name}:", maxsplit=1)
        d[name] = segs[1].split(",", maxsplit=1)[0] if len(segs) > 1 else ""
        if len(segs) <= 1:
            missings[name] += 1
            is_missing = True
    if print_missing and is_missing:
        print(line)
    return d


def compute_accuracy(references, predictions):
    n_correct = defaultdict(int)
    missings = defaultdict(int)
    for ref, pred in zip(references, predictions):
        ref = {
            name: ref.split(f"{name}:", maxsplit=1)[1].split(",", maxsplit=1)[0]
            for name in names
        }
        pred = parse_pred_line(pred, missings, print_missing=True)
        for name in names:
            n_correct[name] += ref[name] == pred[name]
    print(f"Incomplete prediction counts:\n{missings}")
    accuracy = {name: n / NROWS for name, n in n_correct.items()}
    accuracy["all"] = sum(list(accuracy.values())) / len(accuracy)
    return accuracy


def eval(predictions=None):
    if predictions is None:
        predictions = load_lines(f"data/{CHECKPOINT}.txt")
    references = load_lines(f"data/star2000.txt")
    assert len(predictions) == len(references)

    # Eval using accuracy
    accuracy = dict(compute_accuracy(references, predictions))
    print(f"Accuracy:\n{accuracy}")
    return accuracy


if __name__ == "__main__":
    predictions = predict()
    eval(predictions)

    # eval()
