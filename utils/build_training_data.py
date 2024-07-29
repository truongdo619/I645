import json
import random

# Read a jsonl file with utf-8 encoding, save to list of dict
def write_jsonl_utf8(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_jsonl_utf8(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        return [json.loads(line) for line in file]

def sample_by_ratio(data, aug_ratio):
    """
    Selects a subset of the data list based on the given ratio.

    Parameters:
    - data: List from which to sample.
    - ratio: Float representing the ratio of samples to select. 
             If ratio < 1, it will take a fraction of the list.
             If ratio >= 1, it will take at least the number of elements as the list size.

    Returns:
    - A list containing the sampled elements.
    """
    frac_ratio = aug_ratio - int(aug_ratio)
    samples = []
    if int(aug_ratio) > 0:
        samples = random.sample(data, int(aug_ratio))

    remain_samples = set(data) - set(samples)
    if frac_ratio > 0 and random.random() < frac_ratio:
        # Random a number and check if that number is less than the ratio, add one more sample
        samples += random.sample(remain_samples, 1)
    return samples


if __name__ == "__main__":
    augmented_ratios = [0.5, 1, 2, 3, 4, 5]
    train_data = read_jsonl_utf8("./data/train_data.json")
    for augmented_ratio in augmented_ratios:
        print(f"Augmenting training data with ratio {augmented_ratio}...")
        augmented_train_data = []
        for item in train_data:
            org_audio_path = item["audio"]["path"].replace("stop", "stop_augmented")
            augmented_audio_paths = sample_by_ratio([org_audio_path.replace(".wav", f"_aug_{i}.wav") for i in range(5)], augmented_ratio)
            augmented_train_data += [{"audio": {"path": path}, "sentence": item["sentence"], "duration": item["duration"], "logical_form": item["logical_form"]} for path in augmented_audio_paths]
        augmented_train_data = train_data + augmented_train_data
        write_jsonl_utf8(f"./data/train_data_augmented_{augmented_ratio}.json", augmented_train_data)
        print(f"Number of augmented training data with ratio {augmented_ratio}: {len(augmented_train_data)}")

