import torch
from transformers.utils import ContextManagers


def load_tes(
    args,
    text_encoder_classes,
    tokenizers,
    weight_dtype,
    text_encoder_path,
    text_encoder_subfolder,
):
    text_encoder_cls_1, text_encoder_cls_2, text_encoder_cls_3 = text_encoder_classes
    tokenizer_1, tokenizer_2, tokenizer_3 = tokenizers
    text_encoder_1, text_encoder_2, text_encoder_3 = None, None, None
    text_encoder_variant = args.variant

    if args.quantize_method == "bitsandbytes":
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        nf4_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        nf4_config = None

    if tokenizer_1 is not None and not args.model_family == "smoldit":
        if args.model_family.lower() == "pixart_sigma":
            print(
                f"Loading T5-XXL v1.1 text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        elif args.model_family.lower() == "flux":
            print(
                f"Loading OpenAI CLIP-L text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        elif args.model_family.lower() == "kolors":
            print(
                f"Loading ChatGLM language model from {text_encoder_path}/{text_encoder_subfolder}.."
            )
            text_encoder_variant = "fp16"
        elif args.model_family.lower() == "sana":
            print(
                f"Loading Gemma2 language model from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        else:
            print(
                f"Loading CLIP text encoder from {text_encoder_path}/{text_encoder_subfolder}.."
            )
        text_encoder_1 = text_encoder_cls_1.from_pretrained(
            text_encoder_path,
            subfolder=text_encoder_subfolder,
            revision=args.revision,
            quantization_config=nf4_config,
            variant=text_encoder_variant,
            torch_dtype=weight_dtype,
        )
    elif args.model_family.lower() == "smoldit":
        text_encoder_1 = text_encoder_cls_1.from_pretrained(
            "EleutherAI/pile-t5-base",
            torch_dtype=weight_dtype,
            quantization_config=nf4_config,
        ).encoder
        text_encoder_1.eval()

    if tokenizer_2 is not None:
        if args.model_family.lower() == "flux":
            print(
                f"Loading T5 XXL v1.1 text encoder from {args.pretrained_model_name_or_path}/text_encoder_2.."
            )
        else:
            print("Loading LAION OpenCLIP-G/14 text encoder..")
        text_encoder_2 = text_encoder_cls_2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            torch_dtype=weight_dtype,
            quantization_config=nf4_config,
            variant=args.variant,
        )
    if tokenizer_3 is not None and args.model_family == "sd3":
        print("Loading T5-XXL v1.1 text encoder..")
        text_encoder_3 = text_encoder_cls_3.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            torch_dtype=weight_dtype,
            revision=args.revision,
            quantization_config=nf4_config,
            variant=args.variant,
        )

    return text_encoder_variant, text_encoder_1, text_encoder_2, text_encoder_3




def encode_sd3_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
    max_token_length: int = 77,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds




def encode_sd3_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    zero_padding_tokens: bool = True,
    max_sequence_length: int = 77,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    attention_mask = text_inputs.attention_mask.to(device)

    if zero_padding_tokens:
        # for some reason, SAI's reference code doesn't bother to mask the prompt embeddings.
        # this can lead to a problem where the model fails to represent short and long prompts equally well.
        # additionally, the model learns the bias of the prompt embeds' noise.
        return prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
    else:
        return prompt_embeds




def encode_sd3_prompt(
    text_encoders,
    tokenizers,
    prompt,
    is_validation: bool = False,
    zero_padding_tokens: bool = False,
    device=None,
):
    """
    Encode a prompt for an SD3 model.

    Args:
        text_encoders: List of text encoders.
        tokenizers: List of tokenizers.
        prompt: The prompt to encode.
        num_images_per_prompt: The number of images to generate per prompt.
        is_validation: Whether the prompt is for validation. No-op for SD3.

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds).
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    num_images_per_prompt = 1

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = encode_sd3_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = encode_sd3_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        zero_padding_tokens=zero_padding_tokens,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def init_text_encoder(args, logger, device, move_to_accelerator: bool = True):
    from SimpleTuner.helpers.training.text_encoding import (import_model_class_from_model_name_or_path, get_tokenizers)
    from SimpleTuner.helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager
    tokenizer_1, tokenizer_2, tokenizer_3 = get_tokenizers(args)
    text_encoder_cls_1, text_encoder_cls_2, text_encoder_cls_3 = (
        None,
        None,
        None,
    )
    if tokenizer_1 is not None:
        text_encoder_cls_1 = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            args,
            subfolder="text_encoder",
        )
    if tokenizer_2 is not None:
        text_encoder_cls_2 = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            args,
            subfolder="text_encoder_2",
        )
    if tokenizer_3 is not None and args.model_family == "sd3":
        text_encoder_cls_3 = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
            args,
            subfolder="text_encoder_3",
        )
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]
        text_encoder_classes = [
            text_encoder_cls_1,
            text_encoder_cls_2,
            text_encoder_cls_3,
        ]
        (
            text_encoder_variant,
            text_encoder_1,
            text_encoder_2,
            text_encoder_3,
        ) = load_tes(
            args=args,
            text_encoder_classes=text_encoder_classes,
            weight_dtype=torch.float16 if args.mixed_precision=="fp16" else torch.float32,
            tokenizers=tokenizers,
            text_encoder_path=args.pretrained_model_name_or_path,
            text_encoder_subfolder="text_encoder",
        )
    text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]
    tokenizers = [tokenizer_1, tokenizer_2, tokenizer_3]
    if not move_to_accelerator:
        return text_encoders, tokenizers
    else:
        if tokenizer_1 is not None:
            logger.info("Moving text encoder to {}.".format(device))
            text_encoders[0].to(
                device, dtype=torch.float16 if args.mixed_precision=="fp16" else torch.float32
            )
        if tokenizer_2 is not None:
            logger.info("Moving text encoder 2 to {}.".format(device))
            text_encoders[1].to(
                device, dtype=torch.float16 if args.mixed_precision=="fp16" else torch.float32
            )
        if tokenizer_3 is not None:
            logger.info("Moving text encoder 3 to {}.".format(device))
            text_encoders[2].to(
                device, dtype=torch.float16 if args.mixed_precision=="fp16" else torch.float32
            )

        return text_encoders, tokenizers

