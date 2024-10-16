#!/usr/bin/env python
# coding=utf-8
from typing import List, Tuple, Optional, Union, Dict, Optional, Any
import inspect
import argparse
import math
import os
from datetime import datetime
import random
import shutil
from glob import glob
from PIL import Image
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

from tqdm.auto import tqdm, trange


if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

MAX_INFER_BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Train a stable diffusion model.", prog="Train ESD")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, 
        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, 
        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, required=False,
        help="Variant of pretrained model identifier from huggingface.co/models. Provide 'non_ema' for finetuning.")
    
    parser.add_argument("--removing_concepts", type=str, nargs="+", 
        help=("A set of concepts to be removed. "
              "If len == 1 and ends with `.txt` (seperated by newline), read from file."))
    parser.add_argument("--validation_prompts", type=str, nargs="*", default=[],
        help=("A set of prompts evaluated every `--eval_every`. "
              "If len == 1 and ends with `.txt` (seperated by newline), read from file."))
    parser.add_argument("--num_images_per_prompt", type=int, default=1,)
    
    parser.add_argument("--guidance_scale", type=float, default=3.0,
        help="The scale of the CFG guidance for z_t.")
    parser.add_argument("--concept_scale", type=float, default=3.0,
        help="The scale of the safety (negative) guidance for the target.")
    parser.add_argument("--finetuning_method", type=str, default="xattn",
        choices=["full", "selfattn", "xattn", "noxattn", "notime"])

    parser.add_argument("--output_dir", type=str, default="./saved/",
        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--logging_dir", type=str, default="./logs/",
        help="The directory where the logs will be written.")
    parser.add_argument("--image_dir", type=str, default="./images/",
        help="The directory where the images are stored. If not provided, do not save generated images.")
    parser.add_argument("--exp_name", type=str, default="esd")

    parser.add_argument("--log_every", type=int, default=100,
        help="Log the training loss every `--log_every` steps.")
    parser.add_argument("--eval_every", type=int, default=100,
        help="Evaluate the model every `--eval_every` steps.")
    parser.add_argument("--save_every", type=int, default=100,
        help="Save the model every `--save_every` steps.")
    parser.add_argument("--eval_after", type=int, default=0,
        help="Evaluate the model after `--eval_after` steps.")
    parser.add_argument("--eval_at_first", action="store_true",
        help="Evaluate the model at the beginning.")
    parser.add_argument("--max_checkpoints", type=int, default=5,
        help="The maximum number of checkpoints to keep.")
    
    parser.add_argument("--seed", type=int, default=None, required=False,
        help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,
        help="The resolution for input images.")
    parser.add_argument("--train_batch_size", type=int, default=1,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_train_steps", type=int, default=1000,
        help="The total number of training iterations to perform.")
    parser.add_argument("--num_ddpm_steps", type=int, default=1000,
        help="The total number of DDPM steps for training.")
    parser.add_argument("--num_ddim_steps", type=int, default=50,
        help="The total number of DDIM steps for inference.")
    parser.add_argument("--num_inference_steps", type=int, default=25,
        help="The total number of sampling steps for inference.")
    parser.add_argument("--eta", type=float, default=0.0, 
        help="The eta value for DDIM. eta 0.0 corresponds to DDIM, and 1.0 to DDPM.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--ema_decay", type=float, default=0.999,
        help="The decay rate for the exponential moving average model.")

    parser.add_argument("--learning_rate", type=float, default=1e-5,
        help="The initial learning rate (after warmup) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, 
        help="Scale the learning rate by the number GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=("The learning rate scheduler to use. "
              "Choose among `constant`, `linear`, `cosine`, `cosine_warmup`"
              "`cosine_warmup_restart`, `polynomial`, `polynomial_warmup`, `polynomial_warmup_restart`."))
    parser.add_argument("--lr_warmup_steps", type=int, default=500,)
    parser.add_argument("--adam_beta1", type=float, default=0.9,)
    parser.add_argument("--adam_beta2", type=float, default=0.999,)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,)
    parser.add_argument("--weight_decay", type=float, default=1e-4,)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,)

    parser.add_argument("--allow_tf32", action="store_true",
        help="Allow the use of TF32. Only works on certain GPUs.")
    parser.add_argument("--use_fp16", action="store_true", 
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 0])
    
    parser.add_argument("--use_wandb", action="store_true",)
    parser.add_argument("--wandb_project", type=str, default="safe-diffusion")
    
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: torch.nn.Module,
    weight_dtype: torch.dtype,
    step: int,
    device: torch.device,
    prefix: str = "",
):
    logger.info("Running validation...")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # Do not produce more than MAX_INFER_BATCH_SIZE images at a time.
    if args.num_images_per_prompt > MAX_INFER_BATCH_SIZE:
        num_images_per_prompt = MAX_INFER_BATCH_SIZE
        logger.warning(
            f"Reducing the number of images per prompt to {num_images_per_prompt} "
            f"to avoid OOM errors."
        )
        num_iters_per_prompt = math.ceil(args.num_images_per_prompt / num_images_per_prompt)
    else:
        num_images_per_prompt = args.num_images_per_prompt
        num_iters_per_prompt = 1

    if args.image_dir is not None:
        image_dir = args.image_dir
        if step is not None:
            if prefix is None:
                image_folder_name = f"step={step:06d}"
            else:
                image_folder_name = f"step={step:06d}_{prefix}"
            image_dir = os.path.join(image_dir, image_folder_name)
        os.makedirs(image_dir, exist_ok=True)
    else:
        # Do not save images
        image_dir = None

    all_prompts: List[str] = []
    all_images: List[Image.Image] = []
    index = 0
    num_total_images = len(args.validation_prompts) * num_iters_per_prompt
    tbar = trange(num_total_images)
    for i in range(len(args.validation_prompts)):
        tbar.set_description(f"Prompt: {args.validation_prompts[i]}")
        for _ in range(num_iters_per_prompt):
            images = pipeline(
                args.validation_prompts[i],
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                num_images_per_prompt=num_images_per_prompt,
            ).images
            all_images.extend(images)
            all_prompts.extend([args.validation_prompts[i]] * len(images))
            if image_dir is not None:
                for image in images:
                    image.save(os.path.join(image_dir, f"{index:06d}.png"))
                    index += 1
            tbar.update(len(images))
    
    if image_dir is not None:
        with open(os.path.join(image_dir, "prompts.txt"), "w") as f:
            for prompt in all_prompts:
                f.write(prompt + "\n")

    if args.use_wandb:
        wandb.log({
            "val/images": [
                wandb.Image(image, caption=f"{i}: {prompt}")
                for i, (prompt, image) in enumerate(zip(all_prompts, all_images))
            ],
            "step": step,
        })
    
    del pipeline
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def gather_parameters(args: argparse.Namespace, unet: UNet2DConditionModel) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """Gather the parameters to be optimized by the optimizer."""
    names, parameters = [], []
    for name, param in unet.named_parameters():
        if args.finetuning_method == "full":
            # Train all layers.
            names.append(name)
            parameters.append(param)
        elif args.finetuning_method == "selfattn":
            # Attention layer 1 is the self-attention layer.
            if "attn1" in name:
                names.append(name)
                parameters.append(param)
        elif args.finetuning_method == "xattn":
            # Attention layer 2 is the cross-attention layer.
            if "attn2" in name:
                names.append(name)
                parameters.append(param)
        elif args.finetuning_method == "noxattn":
            # Train all layers except the cross attention and time_embedding layers.
            if name.startswith("conv_out.") or ("time_embed" in name):
                # Skip the time_embedding layer.
                continue
            elif "attn2" in name:
                # Skip the cross attention layer.
                continue
            names.append(name)
            parameters.append(param)
        elif args.finetuning_method == "notime":
            # Train all layers except the time_embedding layer.
            if name.startswith("conv_out.") or ("time_embed" in name):
                continue
            names.append(name)
            parameters.append(param)
        else:
            raise ValueError(f"Unknown finetuning method: {args.finetuning_method}")

    return names, parameters


def save_checkpoint(
    args: argparse.Namespace,
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    step: Optional[int]=None,
):
    """Save a checkpoint. If step is None, save the entire pipeline.
    Otherwise, save only the unet model in the folder `step={step}`."""
    max_checkpoints = args.max_checkpoints
    if step is not None:
        output_dir = os.path.join(args.output_dir, f"step={step:06d}")
        # count the number of checkpoints
        if max_checkpoints is not None:
            checkpoints = glob(os.path.join(args.output_dir, "step=*"))
            if len(checkpoints) >= max_checkpoints:
                # sort by step
                checkpoints.sort(key=lambda x: int(x.split("=")[-1]))
                # remove the oldest checkpoint
                shutil.rmtree(checkpoints[0])
                print(f"Removed checkpoint {checkpoints[0]}")
        os.makedirs(output_dir, exist_ok=True)
        unet.save_pretrained(output_dir)
    else:
        output_dir = args.output_dir
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(output_dir)


@torch.no_grad()
def encode_prompt(
    prompt: Union[str, List[str]]=None,
    negative_prompt: Union[str, List[str]]=None,
    removing_prompt: Union[str, List[str]]=None,
    num_images_per_prompt: int=1,
    text_encoder: CLIPTextModel=None,
    tokenizer: CLIPTokenizer=None,
    device: torch.device=None,
):
    """Encode a prompt into a text embedding. Prompt can be None."""
    # Get text embeddings for unconditional and conditional prompts.
    if isinstance(prompt, str):
        prompt = [prompt]
    
    if removing_prompt is not None and isinstance(removing_prompt, str):
        removing_prompt = [removing_prompt]
        assert len(prompt) == len(removing_prompt), f"Safety concept must be the same length as prompt of length {len(prompt)}."
    
    if negative_prompt is not None and isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt]
        assert len(prompt) == len(negative_prompt), f"Negative prompt must be the same length as prompt of length {len(prompt)}."

    batch_size = len(prompt) if prompt is not None else 1

    use_attention_mask = hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask
    device = device if device is not None else text_encoder.device

    # Tokenization
    uncond_input = tokenizer(
        [""] * batch_size if negative_prompt is None else negative_prompt,
        padding="max_length", 
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    if prompt is not None:
        prompt_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
    else:
        prompt_input = None
    
    if removing_prompt is not None:
        removing_input = tokenizer(
            removing_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt",
        )
    else:
        removing_input = None

    # Encoding
    prompt_embeds = text_encoder(
        input_ids=uncond_input["input_ids"].to(device),
        attention_mask=uncond_input["attention_mask"].to(device) if use_attention_mask else None,
    )[0]
    if prompt_input is not None:
        prompt_emb = text_encoder(
            input_ids=prompt_input["input_ids"].to(device),
            attention_mask=prompt_input["attention_mask"].to(device) if use_attention_mask else None,
        )[0]
        prompt_embeds = torch.cat([prompt_embeds, prompt_emb], dim=0)
    
    if removing_input is not None:
        removing_emb = text_encoder(
            input_ids=removing_input["input_ids"].to(device),
            attention_mask=removing_input["attention_mask"].to(device) if use_attention_mask else None,
        )[0]
        prompt_embeds = torch.cat([prompt_embeds, removing_emb], dim=0)

    # Duplicate the embeddings for each image.
    if num_images_per_prompt > 1:
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)
    
    return prompt_embeds


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    
    return extra_step_kwargs


# Sample latents from unet and DDIM scheduler until the given timestep.
@torch.no_grad()
def sample_until(
    until: int,
    latents: torch.Tensor,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
    extra_step_kwargs: Optional[Dict[str, Any]]=None,
):
    """Sample latents until t for a given prompt."""
    timesteps = scheduler.timesteps

    do_guidance = abs(guidance_scale) > 1.0

    # Denoising loop
    for i, t in enumerate(timesteps):
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_guidance
            else latents
        )
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

        # perform guidance
        if do_guidance:
            noise_pred_out = torch.chunk(noise_pred, 2, dim=0)
            noise_pred_uncond, noise_pred_prompt = noise_pred_out[0], noise_pred_out[1]
            # classifier-free guidance term
            cond_guidance = noise_pred_prompt - noise_pred_uncond
            # add the guidance term to the noise residual
            noise_pred = noise_pred_uncond + (guidance_scale * cond_guidance)

        latents = scheduler.step(model_output=noise_pred, timestep=t, sample=latents, **extra_step_kwargs).prev_sample

        if i == (until-1):
            # print(f"Sampled until t={t}, i={i}.")
            break

    return latents


def train_step(
    args: argparse.Namespace,
    prompt: str,
    removing_prompt: str,
    generator: torch.Generator,
    noise_scheduler: DDPMScheduler,
    ddim_scheduler: DDIMScheduler,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet_teacher: UNet2DConditionModel,
    unet_student: UNet2DConditionModel,
    devices: List[torch.device],
) -> torch.Tensor:
    """Train the model a single step for a given prompt and return the loss."""

    unet_student.train()

    # Encode prompt
    prompt_embeds = encode_prompt(
        prompt=prompt, 
        removing_prompt=removing_prompt,
        text_encoder=text_encoder, 
        tokenizer=tokenizer,
        device=devices[1],
    )
    
    uncond_emb, cond_emb, safety_emb = torch.chunk(prompt_embeds, 3, dim=0)
    batch_size = cond_emb.shape[0]

    # Prepare timesteps
    noise_scheduler.set_timesteps(args.num_ddpm_steps, devices[1])

    # Prepare latent codes to generate z_t
    latent_shape = (batch_size, unet_teacher.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=devices[0])
    # Scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma # z_T

    # Normally, DDPM takes 1,000 timesteps for training, and DDIM takes 50 timesteps for inference.
    t_ddim = torch.randint(0, args.num_ddim_steps, (1,))
    t_ddpm_start = round((1 - (int(t_ddim) + 1) / args.num_ddim_steps) * args.num_ddpm_steps)
    t_ddpm_end   = round((1 - int(t_ddim)       / args.num_ddim_steps) * args.num_ddpm_steps)
    t_ddpm = torch.randint(t_ddpm_start, t_ddpm_end, (batch_size,),)
    # print(f"t_ddim: {t_ddim}, t_ddpm: {t_ddpm}")

    # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    extra_step_kwargs = prepare_extra_step_kwargs(noise_scheduler, generator, args.eta)

    with torch.no_grad():
        # args.guidance_scale: s_g in the paper
        prompt_embeds = torch.cat([uncond_emb, cond_emb], dim=0) if args.guidance_scale > 1.0 else uncond_emb
        prompt_embeds = prompt_embeds.to(unet_student.device)

        # Generate latents
        latents = sample_until(
            until=int(t_ddim),
            latents=latents,
            unet=unet_student,
            scheduler=ddim_scheduler,
            prompt_embeds=prompt_embeds,
            guidance_scale=args.guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
        )

        # Stop-grad and send to the second device
        _latents = latents.to(devices[1])
        e_0 = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=uncond_emb).sample
        e_p = unet_teacher(_latents, t_ddpm.to(devices[1]), encoder_hidden_states=safety_emb).sample

        e_0 = e_0.detach().to(devices[0])
        e_p = e_p.detach().to(devices[0])

        # args.concept_scale: s_s in the paper
        noise_target = e_0 - args.concept_scale * (e_p - e_0)

    noise_pred = unet_student(latents, t_ddpm.to(devices[0]), encoder_hidden_states=safety_emb.to(devices[0])).sample

    loss = F.mse_loss(noise_pred, noise_target)
    
    return loss


def main():

    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    args.exp_name = f"{args.exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger.info(f"Experiment name: {args.exp_name}")

    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.logging_dir is not None:
        args.logging_dir = os.path.join(args.logging_dir, args.exp_name)
        os.makedirs(args.logging_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(args.logging_dir, "train.log"),
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )

    if args.image_dir is not None:
        args.image_dir = os.path.join(args.image_dir, args.exp_name)
        os.makedirs(args.image_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, 
            name=args.exp_name, 
            dir=args.logging_dir, 
            config=args,
        )
        args = wandb.config

    logger.info(args)
    
    # You may provide a single file path, or a list of concepts
    if len(args.removing_concepts) == 1 and args.removing_concepts[0].endswith(".txt"):
        with open(args.removing_concepts[0], "r") as f:
            args.removing_concepts = f.read().splitlines()

    if (args.validation_prompts is None) or (len(args.validation_prompts) == 0):
        args.validation_prompts = None
    elif len(args.validation_prompts) == 1 and args.validation_prompts[0].endswith(".txt"):
        with open(args.validation_prompts[0], "r") as f:
            args.validation_prompts = f.read().splitlines()

    # This script requires two CUDA devices
    # Sample latents on the first device, and train the unet on the second device
    devices = [torch.device(f"cuda:{idx}") for idx in args.devices]

    # Load pretrained models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,)
    ddim_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",)

    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
    )
    unet_student = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
    )

    # Freeze vae and text_encoder
    unet_teacher.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.allow_tf32:
        # Allow TF32 on Ampere GPUs to speed up training
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size
        )
    
    names, parameters = gather_parameters(args, unet_student)
    logger.info(f"Finetuning parameters: {names}")
    num_train_param = sum(p.numel() for p in parameters)
    num_total_param = sum(p.numel() for p in unet_student.parameters())
    print(f"Finetuning parameters: {num_train_param} / {num_total_param} ({num_train_param / num_total_param:.2%})")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    lr_scheduler: LambdaLR = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_steps * args.gradient_accumulation_steps,
    )

    # First device -- unet_student, generator
    # Second device -- unet_teacher, vae, text_encoder
    unet_student = unet_student.to(args.devices[0])
    gen = torch.Generator(device=devices[0])

    unet_teacher = unet_teacher.to(devices[1])
    text_encoder = text_encoder.to(devices[1])
    vae = vae.to(args.devices[1])
    if args.seed is not None:
        gen.manual_seed(args.seed)

    if args.use_wandb:
        wandb.watch(unet_student, log="all")
    
    if args.use_fp16:
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler()

    # Set the number of inference time steps
    ddim_scheduler.set_timesteps(args.num_ddim_steps, devices[1])

    # Validation at the beginning
    step = 0
    if args.eval_at_first and (len(args.validation_prompts) > 0):
        validate(
            args=args,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet_teacher,
            weight_dtype=vae.dtype,
            step=step,
            device=devices[1],
        )

    progress_bar = tqdm(range(1, args.num_train_steps+1), desc="Training")

    for step in progress_bar:

        removing_concept = random.choice(args.removing_concepts)
        removing_prompt = removing_concept
        prompt = removing_prompt

        unet_student.train()

        if args.use_fp16:
            with torch.cuda.amp.autocast():
                train_loss = train_step(
                    args=args,
                    prompt=prompt,
                    removing_prompt=removing_prompt,
                    generator=gen,
                    noise_scheduler=noise_scheduler,
                    ddim_scheduler=ddim_scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet_teacher=unet_teacher,
                    unet_student=unet_student,
                    devices=devices,
                )
            
            scaler.scale(train_loss).backward()
            
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    clip_grad_norm_(parameters, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
        
        else:
            train_loss = train_step(
                args=args,
                prompt=prompt,
                removing_prompt=removing_prompt,
                generator=gen,
                noise_scheduler=noise_scheduler,
                ddim_scheduler=ddim_scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet_teacher=unet_teacher,
                unet_student=unet_student,
                devices=devices,
            )
            
            train_loss.backward()
            
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        progress_bar.set_description(f"Training: {train_loss.item():.4f} on c_p: {prompt} - c_s: {removing_concept}")
        if args.use_wandb:
            wandb.log({"train/loss": train_loss.item(), "step": step, "train/lr": lr_scheduler.get_last_lr()[0]})

        if (step % args.log_every == 0) and (args.logging_dir is not None):
            logger.info(f"Step: {step} | Loss: {train_loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.4e}")

        # Validation
        if (step % args.eval_every == 0) and (step >= args.eval_after) and (len(args.validation_prompts) > 0):
            validate(
                args=args,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet_student,
                weight_dtype=vae.dtype,
                step=step,
                device=devices[1],
                prefix="teacher",
            )

            # Save checkpoint
            if step % args.save_every == 0:
                if args.output_dir is not None:
                    save_checkpoint(
                        args=args,
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet_student,
                        step=step,
                    )

    # Save final checkpoint
    if args.output_dir is not None:
        save_checkpoint(
            args=args,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet_student,
        )


if __name__ == "__main__":
    main()