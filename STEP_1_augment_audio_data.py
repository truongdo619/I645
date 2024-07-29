from audiomentations import Compose, AddBackgroundNoise, BandPassFilter, LoudnessNormalization, PitchShift, TimeMask, Mp3Compression, TimeStretch, Shift
import os
import soundfile as sf
import json
from tqdm import tqdm

class AudioAugmenter:
    def __init__(self, noise_dir, augmentation_ratio=1.0, prob=0.125):
        self.noise_dir = noise_dir
        self.augmentation_ratio = augmentation_ratio
        self.augment = Compose([
            AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=10, max_snr_in_db=30, p=prob),
            BandPassFilter(min_center_freq=100.0, max_center_freq=4000.0, p=prob),
            LoudnessNormalization(min_lufs=-31, max_lufs=-13, p=prob),
            PitchShift(min_semitones=-4, max_semitones=4, p=prob),
            TimeMask(min_band_part=0.0, max_band_part=0.2, p=prob),
            Mp3Compression(min_bitrate=16, max_bitrate=64, p=prob),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=prob),
            Shift(min_shift=-0.5, max_shift=0.5, rollover=True, p=prob)
        ])

    def augment_audio(self, audio_file):
        samples, sample_rate = sf.read(audio_file)
        augmented_samples = [self.augment(samples=samples, sample_rate=sample_rate) for _ in range(int(self.augmentation_ratio))]
        base_name = os.path.basename(audio_file).split("/")[-1].split('.')[0]
        save_dir = os.path.dirname(audio_file).replace("stop", "stop_augmented")

        for i, augmented in enumerate(augmented_samples):
            augmented_file = os.path.join(save_dir, f"{base_name}_aug_{i}.wav")
            sf.write(augmented_file, augmented, sample_rate)

# Check if the directory of the file exists or create it recursively
def check_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read a jsonl file with utf-8 encoding, save to list of dict
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    # Usage
    noise_directory = "/home/s2320435/sound_datasets/urbansound8k/audio/fold1"
    train_data = read_jsonl("./data/train_data.json")

    # Check if the save directory exists or create it recursively
    for item in tqdm(train_data):
        check_directory(item["audio"]["path"].replace("stop", "stop_augmented"))

    # Augment the audio files
    augmentation_ratio = 5  # Number of augmentations per original audio
    augmenter = AudioAugmenter(noise_dir=noise_directory, augmentation_ratio=augmentation_ratio)
    for item in tqdm(train_data):
        augmenter.augment_audio(item["audio"]["path"])