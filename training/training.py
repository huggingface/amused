# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple
import argparse
import copy

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel, EMAModel, UVit2DModel, AmusedPipeline, AmusedScheduler
import diffusers.optimization
import torch.nn.functional as F
from diffusers.utils import is_wandb_available

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


class AdapterDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def _image_transform(self, image):
        orig_size = (image.height, image.width)
        image = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)(image)
        # get crop coordinates
        c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(self.size, self.size))
        image = transforms.functional.crop(image, c_top, c_left, self.size, self.size)
        image = transforms.ToTensor()(image)
        crop_coords = (c_top, c_left)
        aes_score = torch.tensor(6.0)
        return image, crop_coords, orig_size, aes_score

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        instance_image = instance_image.resize((self.size, self.size), Image.BILINEAR)
        instance_image, crop_coords, orig_size, aes_score = self._image_transform(instance_image)
        example["image"] = instance_image
        example["orig_size"] = orig_size
        example["crop_coords"] = crop_coords
        example["aesthetic_score"] = aes_score

        text_inputs = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        example["input_ids"] = text_inputs.input_ids[0]
        return example

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="muse_training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    args = parser.parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    return args

def main(args):
    config = OmegaConf.load("/fsx/william/amused/training/config_lora.yaml")

    # Enable TF32 on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        accelerator.init_trackers("amused", config=vars(copy.deepcopy(args)))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        else:
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    is_lora = config.experiment.get("is_lora", False)

    text_encoder = CLIPTextModelWithProjection.from_pretrained(config.model.text_encoder.pretrained, projection_dim=768)
    tokenizer = CLIPTokenizer.from_pretrained(config.model.text_encoder.pretrained)
    if config.model.text_encoder.get("pad_token_id", None):
        tokenizer.pad_token_id = config.model.text_encoder.pad_token_id

    vq_model = VQModel.from_pretrained("openMUSE/diffusers-pipeline", subfolder="vqvae")

    # Freeze the text model and VQGAN
    text_encoder.requires_grad_(False)
    vq_model.requires_grad_(False)

    if is_lora:
        if config.model.get("pretrained_model_path", None) is not None:
            subfolder = config.model.get("pretrained_model_path_subfolder", None)
            model = UVit2DModel.from_pretrained(config.model.pretrained_model_path, subfolder=subfolder)
        else:
            model = UVit2DModel(**config.model.transformer)
    
        if resume_from_checkpoint is not None:
            model = PeftModel.from_pretrained(model, os.path.join(resume_from_checkpoint, "transformer"), is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=['to_q', 'to_k', 'to_v'],
            )
            model = get_peft_model(model, lora_config)
    else:
        if resume_from_checkpoint is not None:
            model = UVit2DModel.from_pretrained(resume_from_checkpoint, subfolder="transformer")
        elif config.model.get("pretrained_model_path", None) is not None:
            subfolder = config.model.get("pretrained_model_path_subfolder", None)
            model = UVit2DModel.from_pretrained(config.model.pretrained_model_path, subfolder=subfolder)
        else:
            model = UVit2DModel(**config.model.transformer)

    model_config = model.config
    
    mask_id = model_config.vocab_size - 1

    # Create EMA
    if config.training.get("use_ema", False):
        if resume_from_checkpoint is not None:
            ema = EMAModel.from_pretrained(os.path.join(resume_from_checkpoint, "ema_model"), model_cls=UVit2DModel)
        else:
            ema = EMAModel(
                model.parameters(),
                decay=config.training.ema_decay,
                update_after_step=config.training.ema_update_after_step,
                model_cls=UVit2DModel,
                model_config=model_config,
            )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            assert len(models) == 1
            models[0].save_pretrained(os.path.join(output_dir, "transformer"))
            weights.pop()

            if config.training.get("use_ema", False):
                ema.save_pretrained(os.path.join(output_dir, "ema_model"))

    def load_model_hook(models, input_dir):
        # All models are initially instantiated from the checkpoint and so
        # don't have to be loaded in the accelerate hook
        assert len(models) == 1
        models.pop()

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate
    if optimizer_config.scale_lr:
        learning_rate = (
            learning_rate
            * config.training.batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    elif optimizer_type == "8bit_adamw":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")
    
    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size = (
        config.training.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = AdapterDataset(
        instance_data_root=dataset_config.instance_data_root,
        instance_prompt=dataset_config.instance_prompt,
        tokenizer=tokenizer,
        size=preproc_config.resolution,
        center_crop=preproc_config.center_crop,
        tokenizer_max_length=preproc_config.max_seq_length,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        collate_fn=default_collate,
    )
    train_dataloader.num_batches = len(train_dataloader)

    lr_scheduler = diffusers.optimization.get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)
    train_dataloader.num_batches = len(train_dataloader)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # TODO: make this configurable
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(device=accelerator.device, dtype=weight_dtype)
    vq_model.to(device=accelerator.device)
    if config.training.get("use_ema", False):
        ema.to(accelerator.device)

    with torch.no_grad():
        empty_input = tokenizer("", padding="max_length", return_tensors="pt").input_ids.to(accelerator.device)
        outputs = text_encoder(empty_input, output_hidden_states=True)
    empty_embeds = outputs.hidden_states[-2]
    empty_clip_embeds = outputs[0]

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = { config.training.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in train_dataloader:
            with torch.no_grad():
                pixel_values, input_ids = batch["image"], batch["input_ids"]
                pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
                input_ids = input_ids.to(accelerator.device, non_blocking=True)
                batch_size = pixel_values.shape[0]

                split_batch_size = config.training.get("split_vae_encode", batch_size)
                # Use a batch of at most split_vae_encode images to encode and then concat the results
                num_splits = math.ceil(batch_size / split_batch_size)
                image_tokens = []
                for i in range(num_splits):
                    start_idx = i * split_batch_size
                    end_idx = min((i + 1) * split_batch_size, batch_size)
                    bs = pixel_values.shape[0]
                    image_tokens.append(
                        vq_model.quantize(vq_model.encode(pixel_values[start_idx:end_idx]).latents)[2][2].reshape(bs, -1)
                    )
                image_tokens = torch.cat(image_tokens, dim=0)

                outputs = text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                encoder_hidden_states = outputs.hidden_states[-2]
                cond_embeds = outputs[0]

                original_sizes = list(map(list, zip(*batch["orig_size"])))
                crop_coords = list(map(list, zip(*batch["crop_coords"])))
    
                aesthetic_scores = batch["aesthetic_score"]
                micro_conds = torch.cat(
                    [torch.tensor(original_sizes).cpu(), torch.tensor(crop_coords).cpu(), aesthetic_scores.unsqueeze(-1).cpu()], dim=-1
                )
    
                micro_conds = micro_conds.to(cond_embeds.device, non_blocking=True)

                batch_size, seq_len = image_tokens.shape

                timesteps = torch.rand(batch_size, device=image_tokens.device)
                mask_prob = torch.cos(timesteps * math.pi * 0.5)
                mask_prob = mask_prob.clip(config.training.min_masking_rate)

                num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
                batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)

                input_ids = torch.where(mask, mask_id, image_tokens)
                labels = torch.where(mask, image_tokens, -100)

            # Train Step
            with accelerator.accumulate(model):
                if config.training.cond_dropout_prob > 0.0:
                    assert encoder_hidden_states is not None

                    batch_size = encoder_hidden_states.shape[0]

                    mask = (
                        torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                        < config.training.cond_dropout_prob
                    )

                    empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                    encoder_hidden_states = torch.where(
                        (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                    )

                    empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                    cond_embeds = torch.where((cond_embeds * mask.squeeze(-1)).bool(), cond_embeds, empty_clip_embeds_)

                bs = input_ids.shape[0]
                vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
                resolution = config.dataset.preprocessing.resolution // vae_scale_factor
                input_ids = input_ids.reshape(bs, resolution, resolution)

                logits = model(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    micro_conds=micro_conds,
                    pooled_text_emb=cond_embeds,
                ).reshape(bs, model_config.codebook_size, -1).permute(0, 2, 1).reshape(-1, model_config.codebook_size)

                loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.0,
                    reduction="mean",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size)).mean()

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.training.get("use_ema", False):
                    ema.step(model.parameters())

                if (global_step + 1) % config.experiment.log_every == 0:
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(args, config, accelerator, global_step + 1)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    if config.training.get("use_ema", False):
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    with torch.no_grad():
                        logger.info("Generating images...")
                        model.eval()

                        # TODO load properly
                        scheduler = AmusedScheduler.from_pretrained("openMUSE/diffusers-pipeline", subfolder="scheduler")

                        pipe = AmusedPipeline(transformer=accelerator.unwrap_model(model), tokenizer=tokenizer, text_encoder=text_encoder, vqvae=vq_model, scheduler=scheduler)
                        pipe.set_progress_bar_config(disable=True)

                        with open(config.dataset.params.validation_prompts_file, "r") as f:
                            validation_prompts = f.read().splitlines()

                        pil_images = pipe(prompt=validation_prompts).images

                        wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
                        wandb.log({"generated_images": wandb_images}, step=global_step+1)

                        model.train()

                    if config.training.get("use_ema", False):
                        ema.restore(model.parameters())

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.get("use_ema", False):
            ema.copy_to(model.parameters())
        model.save_pretrained(args.output_dir)

    accelerator.end_training()


def save_checkpoint(args, config, accelerator, global_step):
    output_dir = args.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main(parse_args())