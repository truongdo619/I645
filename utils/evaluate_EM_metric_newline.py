import json
import argparse
from tqdm import tqdm
import ast

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prediction", type=str, default="output_full_multi_task/whisper-base/checkpoint-best/predictions.json", help="Path to the predicted json file")
parser.add_argument("--is_full_form", action='store_true', help="Whether the prediction is in full form")
parser.add_argument("--pred_is_dict", action='store_false', help="Whether the prediction is in decoupled form")
parser.add_argument("--gold_is_dict", action='store_false', help="Whether the prediction is in decoupled form")
parser.add_argument("--sep_token", type=str, default="\n", help="Whether the prediction is in decoupled form")
parser.add_argument("--start_chunk", type=int, default=None, help="Start chunk")
parser.add_argument("--num_chunks", type=int, default=None, help="Num chunks")

args = parser.parse_args()
print(args)

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

OPEN_BRACKET = "["
CLOSE_BRACKET = "]"
def convert_full_form_to_decoupled_form(full_form):
    tokens = full_form.split(" ")
    unexpected_token_num_check = 1
    cur_num_check = 0
    result = []
    for token in tokens:
        if token.startswith(OPEN_BRACKET) or token.startswith(CLOSE_BRACKET) or cur_num_check != unexpected_token_num_check:
            result.append(token)
        if token.startswith(OPEN_BRACKET):
            cur_num_check += 1
        if token.startswith(CLOSE_BRACKET):
            cur_num_check -= 1
    return " ".join(result)


def calculate_wer(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
	total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer

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
    # EM = lower_predicted_tokens == lower_gold_tokens or predicted_tokens == gold_tokens
    EM = predicted.lower().replace(" ", "") == gold.lower().replace(" ", "")
    # EM Tree Metric
    predicted_schema_tokens = extract_schema(predicted)
    gold_schema_tokens = extract_schema(gold)
    EM_tree = predicted_schema_tokens == gold_schema_tokens
    return EM, EM_tree


EM_correct, EM_tree_correct, total = 0, 0, 0
if args.start_chunk is not None and args.num_chunks is not None:
    data = []
    for chunk in range(args.start_chunk, args.num_chunks+1):
        prefix = args.prediction.split(".json")[0]
        filename = f"{prefix}__chunk_{chunk}_per_{args.num_chunks}.json"
        data += read_json(filename)
else:
    data = read_json(args.prediction)

slu_wer = 0
total_wer = 0
for item in tqdm(data):
    # try:
        if args.pred_is_dict:
            pred_dict = ast.literal_eval(item["predicted"])
            pred = pred_dict["Intent"] 
        else:
            pred = item["predicted"].split(args.sep_token)[-1].strip()
            pred_asr = item["predicted"].split(args.sep_token)[0].strip()
        if args.is_full_form:
            pred = convert_full_form_to_decoupled_form(pred)
        
        if args.gold_is_dict:
            gold_dict = ast.literal_eval(item["gold"])
            gold = gold_dict["Intent"]
        else:
            gold = item["gold"].split(args.sep_token)[-1].strip()
            gold_asr = item["gold"].split(args.sep_token)[0].strip()
            if not (gold.startswith("Intent:") or gold.startswith("<|intent|>") or gold.startswith("[IN:")):
                continue

        if args.is_full_form:
            gold = convert_full_form_to_decoupled_form(gold)

        cur_result = compare(pred, gold)
        EM_correct += cur_result[0]
        EM_tree_correct += cur_result[1]
        # wer_rate = calculate_wer(pred_asr, gold_asr)
        if item["predicted"] != "Error!":
            # slu_wer += wer_rate
            total_wer += 1
        if not cur_result[0]:
            print("===============")
            print(pred)
            print(gold)
    # except:
    #     pass
        total += 1


print(f"EM: {EM_correct}/{total} = {EM_correct/total}")
print(f"EM Tree: {EM_tree_correct}/{total} = {EM_tree_correct/total}")
# print(f"SLU WER: {slu_wer/total_wer}")