import argparse
import functools
import gc
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments


def extract_schema(tree_str):
    tokens = tree_str.split(" ")
    result = []
    for token in tokens:
        if token.startswith("[") or token.startswith("]"):
            result.append(token)
    return result


def compare(predicted, gold):
    # EM Metric
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    lower_predicted_tokens = predicted.lower().split()
    lower_gold_tokens = gold.lower().split()
    EM = lower_predicted_tokens == lower_gold_tokens or predicted_tokens == gold_tokens
    
    # EM Tree Metric
    predicted_schema_tokens = extract_schema(predicted)
    gold_schema_tokens = extract_schema(gold)
    EM_tree = predicted_schema_tokens == gold_schema_tokens
    return EM, EM_tree

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test_lower_text_label_aligned.json",            help="Path of the test set")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="Path of the merged model, or the name of the model on huggingface")
add_arg("batch_size",  type=int, default=16,        help="Batch size for evaluation")
add_arg("num_workers", type=int, default=8,         help="Number of threads for data reading")
add_arg("language",    type=str, default=None, help="Set the language, can be full name or abbreviation, if None then it is multilingual")
add_arg("timestamps",  type=bool, default=False,    help="Whether to use timestamp data during evaluation")
add_arg("min_audio_len",     type=float, default=0.5,  help="Minimum audio length, in seconds")
add_arg("max_audio_len",     type=float, default=30,   help="Maximum audio length, in seconds")
add_arg("local_files_only",  type=bool,  default=True, help="Whether to only load the model locally, do not try to download")
add_arg("task",       type=str, default=None, help="Task of the model")
add_arg("en_language_generation",  type=bool,  default=False, help="English in generation")
add_arg("start_index",  type=int,  default=2, help="Start Index")
add_arg("num_beam",  type=int,  default=1, help="Start Index")
args = parser.parse_args()
print_arguments(args)

# Check if the model path is valid
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"The model file {args.model_path} does not exist, please check whether the model has been successfully merged, or whether it is a model existing on huggingface"
# Get the data processor of Whisper, which includes feature extractor, tokenizer

print(args.model_path)
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

forced_decoder_ids = processor.get_decoder_prompt_ids()
print(forced_decoder_ids)
# forced_decoder_ids = None

# Get the model
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.eval()

# Get the test data
test_dataset = CustomDataset(data_list_path=args.test_data,
                             language=args.language,
                             processor=processor,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)



# Data padding
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# Start evaluation
total = 0
EM_correct = 0
EM_tree_correct = 0
result = []
if args.language is not None and args.task is not None:
    args.start_index = 4


for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.no_grad():
        if not args.en_language_generation:
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :args.start_index].cuda(),
                    max_new_tokens=255).cpu().numpy())
        else:
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :args.start_index].cuda(),
                    max_new_tokens=255, language="en").cpu().numpy())
        labels = batch["labels"].cpu().numpy()
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        # Convert the predicted and actual tokens to text
        decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # print(batch["labels"][:, :2].cuda())
        # print(generated_tokens)
        # print(batch["labels"])
        # print(processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False))
        for pred, gold in zip(decoded_preds, decoded_labels):
            tmp = {}
            tmp["predicted"] = pred
            tmp["gold"] = gold
            result.append(tmp)
            print("===================")
            print(f"Predicted: {pred}")
            print(f"Gold: {gold}")

write_json(result, os.path.join(args.model_path, "predictions.json"))