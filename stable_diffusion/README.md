# QZO Stable Diffusion 3.5 fine-tuning

We provide an example of how to fine-tune a Stable Diffusion 3.5 model on a custom dataset using the [train_text_to_image.py](./QZO/stable_diffusion/text_to_image/train_text_to_image.py) with QZO. This script is adapted from the original [🤗Diffusers Example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) incorporating functions from the [SimpleTuner](https://github.com/bghira/SimpleTuner) through relative imports.

## Setup

1. **Clone the SimpleTuner repository**  
   Create a folder named `SimpleTuner` under `QZO/stable_diffusion` and clone the repository:
   ```bash
   mkdir /QZO/stable_diffusion/SimpleTuner
   git clone https://github.com/bghira/SimpleTuner /QZO/stable_diffusion/SimpleTuner

2. **Install dependencies**  
   Navigate into the `stable_diffusion` folder and install the required packages:
   ```bash
   cd /QZO/stable_diffusion
   pip install -r requirements.txt

3. **Configure Accelerate**
   ```bash
   accelerate config

## Styles example
We demonstrate fine-tuning Stable Diffusion 3.5 Large using the [Styles Dataset](https://huggingface.co/datasets/rezashkv/styles), which contains multiple artistic styles. This example focuses on creating Tarot style images.

1. Prepare Tarot Style Dataset
   ```python
   from datasets import load_dataset

   # Load Styles dataset
   dataset = load_dataset("rezashkv/styles")

   # Filter for Tarot style samples
   def is_tarot(example):
       return example["style"] == "tarot"

   filtered_dataset = dataset.filter(is_tarot)

   # Save processed dataset
   filtered_dataset.save_to_disk("tarot_dataset")
   ```
2. Launch Fine-tuning
   ```bash
   bash DATASET="tarot_dataset" stable_diffusion3.5l_QZO_4bit.sh

## Acknowledgment

This code is builds upon these open-source works & repos:
- [🤗 Diffusers](https://github.com/huggingface/diffusers)
- [MeZO](https://github.com/ModalMinds/MM-EUREKA)
- [SimpleTuner](https://github.com/bghira/SimpleTuner)
- [sd3.5](https://github.com/Stability-AI/sd3.5)
