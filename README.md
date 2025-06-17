<h1 align="center">QZO: Fine-tuning Quantized Neural Networks with
Zeroth-order Optimization</h1>
<p align="center"><i>A novel memory-efficient training method that reduces the total VRAM cost by more than 18x.</i></p>

<p align="center">
          🤗 <a href="https://huggingface.co/papers/2505.13430">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/pdf/2505.13430">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="https://www.maifoundations.com/blog/qzo/">Blog</a> &nbsp&nbsp 
</p>

This is the official code implementation of the paper 'Fine-tuning Quantized Neural Networks with
Zeroth-order Optimization'.

# 📰News
* **`[2025/06/17]`:**🔥**We have released our code.**
* **`[2025/05/20]`:**🔥**We have released our paper on [arXiv](https://arxiv.org/pdf/2505.13430).**

# ✈️Introduction
Fine-tuning large language models (LLMs) unlocks their potential for various downstream tasks. However, as the parameter size of LLMs continues to grow exponentially, a significant bottleneck in GPU memory becomes a major issue. In this work, we propose a novel memory-efficient training method, Quantized Zeroth-order Optimization (QZO), which minimizes the VRAM cost of model weights, gradients, and optimizer states within a unified work. Notably, QZO achieves a total VRAM reduction of 18x compared with regular fine-tuning during the memory profiling (see the figure below).

*Currently, this repository contains only the code implementation for large language models. The rest of the code related to stable diffusion will be released later.*  

![](.\images\memory_profiling.jpg)

# 🗒️Installation

1. Clone the repository and enter its root folder

   ```shell
   git clone https://github.com/maifoundations/QZO.git
   cd QZO
   ```

2. Create a CONDA environment and install the required packages

   ```shell
   conda create -n qzo --python==3.12.0
   conda activate qzo
   pip install -r requirements.txt
   ```

3. To start training, you may refer to the example scripts (`scripts/examples.sh`) located in both the `large_language_models` and `stable_diffusion` folders. 

   You may also need to comment out part of the sanity check codes in `transformers.trainer.py` to support the direct fine-tuning of a quantized language model. For example, if you are using `transformers==4.48.0`, you need to comment out the following code starting from line 553 in the trainer scripts, and append a `pass` in the end:

   ```python
   if _is_quantized_and_base_model and not _is_peft_model(model) and not _is_model_quantized_and_qat_trainable:
       # raise ValueError(
       #     "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
       #     " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
       #     " for more details"
       # )
       pass
   ```


# 🎓Citation
```
@article{shang2025fine,
  title={Fine-tuning Quantized Neural Networks with Zeroth-order Optimization},
  author={Shang, Sifeng and Zhou, Jiayi and Lin, Chenyu and Li, Minxian and Zhou, Kaiyang},
  journal={arXiv preprint arXiv:2505.13430},
  year={2025}
}
```
# Acknowledgment
Our code is built upon the following projects: [MeZO](https://github.com/ModalMinds/MM-EUREKA), [AQLM](https://github.com/Vahe1994/AQLM).

