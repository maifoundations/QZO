# MeZO
CUDA_VISIBLE_DEVICES=0 MODEL=facebook/opt-6.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/mezo.sh

# GPTQ Quantization
CUDA_VISIBLE_DEVICES=0 python quantization.py --model_path facebook/opt-6.7b --quant_mode gptq --quant_path quantized

# QZO
CUDA_VISIBLE_DEVICES=0 MODEL=quantized/opt-6.7b-gptq-b4-g128 TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash scripts/qzo.sh