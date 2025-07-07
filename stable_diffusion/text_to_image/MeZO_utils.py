import torch
import torch.nn.functional as F
import numpy as np

############## MeZO ##############
def zo_perturb_parameters(args, named_parameters_to_optim, lr, random_seed=None, scaling_factor=1):
    """
    Perturb the parameters with random vector z.
    Input:
    - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
    - scaling_factor: theta = theta + scaling_factor * z * eps
    """

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(random_seed if random_seed is not None else 0)

    for name, param in named_parameters_to_optim:
        if not args.quantized_model_finetune:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * args.zo_eps
        else:
            if args.quantize_method=="torchao":
                #only finetune quantization scale, not scale & zero_point
                z = torch.normal(mean=0, std=1, size=param.tensor_impl.scale.size(), device=param.tensor_impl.scale.device, dtype=param.tensor_impl.scale.dtype)
                param.tensor_impl.scale = param.tensor_impl.scale + scaling_factor * z * args.zo_eps
            elif args.quantize_method=="bitsandbytes":
                z = torch.normal(mean=0, std=1, size=param.quant_state.absmax.size(),
                                 device=param.quant_state.absmax.device, dtype=param.quant_state.absmax.dtype)
                param.quant_state.absmax = param.quant_state.absmax + scaling_factor * z * args.zo_eps


def zo_forward(model, batch, args, timesteps, target, device, encoder_hidden_states=None):
    """
    Get (no gradient) loss from the model. Dropout is turned off too.
    """
    model.eval()

    with torch.inference_mode():
        if args.model_family == 'sd1':
            model_pred = model(batch["noisy_latents"], timesteps, encoder_hidden_states, return_dict=False)[0]
        elif args.model_family == 'sd3':
            model_pred = model(hidden_states=batch["noisy_latents"], timestep=timesteps, encoder_hidden_states=batch['prompt_embeds'].to(device),pooled_projections=batch["pooled_prompt_embeds"].to(device),return_dict=False)[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        if args.local_rank !=-1:
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            loss += avg_loss.item() / args.gradient_accumulation_steps #copy from train_text_to_image, not tested
    return loss.detach()


def zo_step(model, batch, args, timesteps, target, lr, device, encoder_hidden_states=None):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """
    # What parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if not args.quantized_model_finetune:
            if param.requires_grad:
                named_parameters_to_optim.append((name, param))
        else:
            if args.quantize_method=="torchao":
                #parameter: quantization scale
                #!notice!!! torchao quantize the model and set params.requires_grad=False
                if hasattr(param, 'tensor_impl'):
                    named_parameters_to_optim.append((name, param))
            elif args.quantize_method=="bitsandbytes" and hasattr(param, 'bnb_quantized'):
                if param.bnb_quantized:
                    named_parameters_to_optim.append((name, param))


    # Sample the random seed for sampling z
    zo_random_seed = np.random.randint(1000000000)

    # First function evaluation
    zo_perturb_parameters(args, named_parameters_to_optim, lr=lr, random_seed=zo_random_seed, scaling_factor=1)
    loss1 = zo_forward(model, batch, args, timesteps, target, device, encoder_hidden_states=encoder_hidden_states)

    # Second function evaluation
    zo_perturb_parameters(args, named_parameters_to_optim, lr=lr, random_seed=zo_random_seed, scaling_factor=-2)
    loss2 = zo_forward(model, batch, args, timesteps, target, device, encoder_hidden_states=encoder_hidden_states)

    projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()

    # grad_clip
    if projected_grad > args.grad_clip_threshold:
        projected_grad = args.grad_clip_threshold
    elif projected_grad < -args.grad_clip_threshold:
        projected_grad = -args.grad_clip_threshold

    # No gradient accumulation support
    assert args.gradient_accumulation_steps == 1

    # Reset model back to its parameters at start of step
    zo_perturb_parameters(args, named_parameters_to_optim, lr=lr, random_seed=zo_random_seed, scaling_factor=1)

    return loss1, projected_grad, zo_random_seed, named_parameters_to_optim

def zo_update(args, named_parameters_to_optim, zo_random_seed, projected_grad, lr):
    """
    Update the parameters with the estimated gradients.
    """

    # Reset the random seed for sampling zs
    torch.manual_seed(zo_random_seed)

    for name, param in named_parameters_to_optim:
        # Resample z
        if not args.quantized_model_finetune:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        else:
            if args.quantize_method=="torchao":
                z = torch.normal(mean=0, std=1, size=param.tensor_impl.scale.size(), device=param.tensor_impl.scale.device,
                             dtype=param.tensor_impl.scale.dtype)
            elif args.quantize_method=="bitsandbytes":
                z = torch.normal(mean=0, std=1, size=param.quant_state.absmax.size(), device=param.quant_state.absmax.device,
                             dtype=param.quant_state.absmax.dtype)
        weight_decay = 0 #default in transformers.TrainingArguments
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            if not args.quantized_model_finetune:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                if args.quantize_method=="torchao":
                    param.tensor_impl.scale = param.tensor_impl.scale - lr * (projected_grad * z + weight_decay * param.tensor_impl.scale)
                elif args.quantize_method=="bitsandbytes":
                    param.quant_state.absmax = param.quant_state.absmax - lr * (projected_grad * z + weight_decay * param.quant_state.absmax)
        else:
            if not args.quantized_model_finetune:
                param.data = param.data - lr * (projected_grad * z)
            else:
                if args.quantize_method=="torchao":
                    param.tensor_impl.scale = param.tensor_impl.scale - lr * (projected_grad * z)
                elif args.quantize_method=="bitsandbytes":
                    param.quant_state.absmax = param.quant_state.absmax - lr * (projected_grad * z)