# [JAIST-I645] Perceptual Audio Augmentation Techniques for Enhanced Spoken Language Understanding

Spoken Language Understanding (SLU) aims to interpret and extract meaningful information from spoken language inputs. This repository provides a framework for applying perceptual audio augmentation techniques to enhance SLU models.

## Usage

### 1. Set up the environment

To set up the environment, use the following commands:

```bash
conda create --name i645 python=3.8
conda activate i645
pip install -e .
pip install audiomentations
```

### 2. Running the source code

This framework consists of three main steps, which should be executed sequentially.

#### Step 1: Augment audio data

In this step, we use the [audiomentations library](https://github.com/iver56/audiomentations) for perceptual audio augmentation. Start by downloading the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html) to use as background noise for the augmentation. Then, download the main STOP dataset and update the paths in the `data/train_data.json`, `data/eval_data.json`, and `data/test_data.json` files according to the location of the STOP dataset. Once done, run the following command to start augmenting the audio data:

```bash
python STEP_1_augment_audio_data.py
```

#### Step 2: Build final training data

Next, combine the original training data with the augmented data to create the final training dataset. Use the following command:

```bash
python STEP_2_build_training_data.py
```

#### Step 3: Train SLU model

With the augmented training data ready, proceed to train the SLU model. The training can be executed for different settings using the scripts provided in the `scripts` folder. Run the following commands:

```bash
bash scripts/train_and_inference.sh
bash scripts/train_and_inference_with_augmented_ratio_50.sh
```
