'''
Quantize LLMs with methods supported by HuggingFace transformers

Coding Reference:
autoawq:
https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY#scrollTo=Qn_P_E5p7gAN

gptqmodel
https://github.com/ModelCloud/GPTQModel
'''

import os
import os.path as osp
import argparse

import torch
import logging
from typing import List, Union
from datasets import load_dataset
from transformers import AutoTokenizer, AwqConfig, AutoConfig
from gptqmodel import GPTQModel, QuantizeConfig

# cloned from AWQ repository:
# https://github.com/mit-han-lab/llm-awq/blob/main/awq/utils/calib_data.py

def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]


def GPTQ(args):
    model_path = args.model_path
    quant_path = args.quant_path
    
    # For GPTQ, we utilize PilEval datset for calibration
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    calibration_dataset = get_calib_dataset(data="pileval", tokenizer=tokenizer)

    # Preprocess
    processed_calibration_dataset = []
    for idx in range(len(calibration_dataset)):
        processed_calibration_dataset.append(calibration_dataset[idx][0].tolist())

    quant_config = QuantizeConfig(bits=4, group_size=128) # no zeropoint
    model = GPTQModel.from_pretrained(model_path, quant_config, trust_remote_code=True, device_map='auto')

    model.quantize(processed_calibration_dataset, tokenizer=tokenizer, batch_size=2, calibration_enable_gpu_cache=True)
    
    # save model weights
    model_name = model_path.split('/')[-1]
    suffix = '-gptq' + '-b4' + '-g128'
    quant_path = osp.join(args.quant_path, model_name+suffix)
    model.save(quant_path)
    return

def main(args):
    if args.quant_mode == 'gptq':
        GPTQ(args)
    else:
        raise NotImplementedError
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantization with Transformers')
    parser.add_argument('--model_path', type=str, default='facebook/opt-125m')
    parser.add_argument('--quant_mode', type=str, choices=['gptq'], default='gptq')
    parser.add_argument('--quant_path', type=str, default='quantized')
    # parser.add_argument('--zero_point', type=bool, action='store_true', default=True)
    args = parser.parse_args()
    main(args)