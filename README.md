# amused

![collage](./assets/collage_small.png)
<sup><sub>Images cherry-picked from 512 and 256 models. Images are degraded to load faster. See ./assets/collage_full.png for originals</sub></sup>

[[Paper]]()
[[Models]]()
[[Colab]]()
[[Training Code]]()

| Model | Params |
|-------|--------|
| [amused-256](https://huggingface.co/openMUSE/diffusers-pipeline-256-finetuned) | 603M |
| [amused-512](https://huggingface.co/openMUSE/diffusers-pipeline) | 608M |

TODO - checkpoints

TODO - why/where to use amused

## 1. Usage

### Text to image

#### 256x256 model

```python
import torch
from diffusers import AmusedPipeline

pipe = AmusedPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline-256-finetuned", torch_dtype=torch.float16
)  # TODO - fix path
pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans in fp16
pipe = pipe.to("cuda")

prompt = "cowboy"
image = pipe(prompt, generator=torch.Generator('cuda').manual_seed(8)).images[0]
image.save('text2image_256.png')
```

![text2image_256](./assets/text2image_256.png)

#### 512x512 model

```python
import torch
from diffusers import AmusedPipeline

pipe = AmusedPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline", torch_dtype=torch.float16
)  # TODO - fix path
pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans n fp16
pipe = pipe.to("cuda")

prompt = "summer in the mountains"
image = pipe(prompt, generator=torch.Generator('cuda').manual_seed(2)).images[0]
image.save('text2image_512.png')
```

![text2image_512](./assets/text2image_512.png)

### Image to image

#### 256x256 model

```python
import torch
from diffusers import AmusedImg2ImgPipeline
from diffusers.utils import load_image

pipe = AmusedImg2ImgPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline-256-finetuned", torch_dtype=torch.float16
)  # TODO - fix path
pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans in fp16
pipe = pipe.to("cuda")

prompt = "apple watercolor"
input_image = (
    load_image(
        "https://raw.githubusercontent.com/huggingface/amused/main/assets/image2image_256_orig.png"
    )
    .resize((256, 256))
    .convert("RGB")
)

image = pipe(prompt, input_image, strength=0.7, generator=torch.Generator('cuda').manual_seed(3)).images[0]
image.save('image2image_256.png')
```

![image2image_256_orig](./assets/image2image_256_orig.png) ![image2image_256](./assets/image2image_256.png)

#### 512x512 model

```python
import torch
from diffusers import AmusedImg2ImgPipeline
from diffusers.utils import load_image

pipe = AmusedImg2ImgPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline", torch_dtype=torch.float16
)  # TODO - fix path
pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans in fp16
pipe = pipe.to("cuda")

prompt = "winter mountains"
input_image = (
    load_image(
        "https://raw.githubusercontent.com/huggingface/amused/main/assets/image2image_512_orig.png"
    )
    .resize((512, 512))
    .convert("RGB")
)

image = pipe(prompt, input_image, generator=torch.Generator('cuda').manual_seed(15)).images[0]
image.save('image2image_512.png')
```

![image2image_512_orig](./assets/image2image_512_orig.png) ![image2image_512](./assets/image2image_512.png)

### Inpainting

#### 256x256 model

```python
import torch
from diffusers import AmusedInpaintPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = AmusedInpaintPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline-256-finetuned", torch_dtype=torch.float16
)  # TODO - fix path
pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans with this example when in fp16
pipe = pipe.to("cuda")

prompt = "a man with glasses"
input_image = (
    load_image(
        "https://raw.githubusercontent.com/huggingface/amused/main/assets/inpainting_256_orig.png"
    )
    .resize((256, 256))
    .convert("RGB")
)
mask = (
    load_image(
        "https://raw.githubusercontent.com/huggingface/amused/main/assets/inpainting_256_mask.png"
    )
    .resize((256, 256))
    .convert("L")
)    

for seed in range(20):
    image = pipe(prompt, input_image, mask, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    image.save(f'inpainting_256_{seed}.png')

```

![inpainting_256_orig](./assets/inpainting_256_orig.png) ![inpainting_256_mask](./assets/inpainting_256_mask.png) ![inpainting_256](./assets/inpainting_256.png)

#### 512x512 model

```python
import torch
from diffusers import AmusedInpaintPipeline
from diffusers.utils import load_image

pipe = AmusedInpaintPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline", torch_dtype=torch.float16
)  # TODO - fix path
pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans with this example when in fp16
pipe = pipe.to("cuda")

prompt = "fall mountains"
input_image = (
    load_image(
        "https://raw.githubusercontent.com/huggingface/amused/main/assets/inpainting_512_orig.jpeg"
    )
    .resize((512, 512))
    .convert("RGB")
)
mask = (
    load_image(
        "https://raw.githubusercontent.com/huggingface/amused/main/assets/inpainting_512_mask.png"
    )
    .resize((512, 512))
    .convert("L")
)
image = pipe(prompt, input_image, mask, generator=torch.Generator('cuda').manual_seed(0)).images[0]
image.save('inpainting_512.png')
```

![inpainting_512_orig](./assets/inpainting_512_orig.jpeg) 
![inpainting_512_mask](./assets/inpainting_512_mask.png) 
![inpainting_512](./assets/inpainting_512.png)

## 2. Performance

Amused inherits performance benefits from original [muse](https://arxiv.org/pdf/2301.00704.pdf). 

1. Parallel decoding: The model follows a denoising schedule that aims to unmask some percent of tokens at each denoising step. At each step, all masked tokens are predicted, and some number of tokens that the network is most confident about are unmasked. Because multiple tokens are predicted at once, we can generate a full 256x256 or 512x512 image in around 12 steps. In comparison, an autoregressive model must predict a single token at a time. Note that a 256x256 image with the 16x downsampled VAE that muse uses will have 256 tokens.

2. Fewer sampling steps: Compared to many diffusion models, muse requires fewer samples.

Additionally, amused uses the smaller CLIP as its text encoder instead of T5 compared to muse. Amused is also smaller with ~600M params compared the largest 3B param muse model. Note that being smaller, amused produces comparably lower quality results.

![a100_bs_1](./assets/a100_bs_1.png)
![a100_bs_8](./assets/a100_bs_8.png)
![4090_bs_1](./assets/4090_bs_1.png)
![4090_bs_8](./assets/4090_bs_8.png)

### Muse performance knobs

|                     | Uncompiled Transformer + regular attention | Uncompiled Transformer + flash attention (ms) | Compiled Transformer (ms) | Speed Up |
|---------------------|--------------------------------------------|-------------------------|----------------------|----------|
| 256 Batch Size 1    |                594.7                      |         507.7                |    212.1                  |   58%       |
| 512 Batch Size 1    |                637                      |        547                 |       249.9               |     54%     |
| 256 Batch Size 8    |                719                      |        628.6                 |        427.8              |    32%      |
| 512 Batch Size 8    |                  1000                    |         917.7                |       703.6               |    23%      |

Flash attention is enabled by default in the diffusers codebase through torch `F.scaled_dot_product_attention`

### torch.compile
To use torch.compile, simply wrap the transformer in torch.compile i.e.

```python
pipe.transformer = torch.compile(pipe.transformer)
```

Full snippet:

```python
import torch
from diffusers import AmusedPipeline

pipe = AmusedPipeline.from_pretrained(
    "openMUSE/diffusers-pipeline-256-finetuned", torch_dtype=torch.float16
)  # TODO - fix path

# HERE use torch.compile
pipe.transformer = torch.compile(pipe.transformer)

pipe.vqvae.to(torch.float32)  # TODO - vqvae is producing nans in fp16
pipe = pipe.to("cuda")

prompt = "cowboy"
image = pipe(prompt, generator=torch.Generator('cuda').manual_seed(8)).images[0]
image.save('text2image_256.png')
```

## 3. Training

Amused can be finetuned on simple datasets relatively cheaply and quickly. Using 8bit optimizers, lora, and gradient accumulation, amused can be finetuned with as little as 5.5 GB. Here are a set of examples for finetuning amused on some relatively simple datasets. These training recipies are aggressively oriented towards minimal resources and fast verification -- i.e. the batch sizes are quite low and the learning rates are quite high. For optimal quality, you will probably want to increase the batch sizes and decrease learning rates.

All training examples use fp16 mixed precision and gradient checkpointing. We don't show 8 bit adam + lora as its about the same memory use as just using lora (bitsandbytes uses full precision optimizer states for weights below a minimum size).

### Finetuning the 256 checkpoint

These examples finetune on this [nouns](https://huggingface.co/datasets/m1guelpf/nouns) dataset.

Example results:

![noun1](./assets/noun1.png) ![noun2](./assets/noun2.png) ![noun3](./assets/noun3.png)

#### Full finetuning

Batch size: 8, Learning rate: 1e-4, Gives decent results in 750-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      19.7 GB       |
|    4        |          2                   |     8             |      18.3 GB       |
|    1        |          8                   |     8             |      17.9 GB       |

```sh
# TODO - update model path
accelerate launch training/training.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 1e-4 \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline-256-finetuned \
    --instance_data_dataset  'm1guelpf/nouns' \
    --image_key image \
    --prompt_key text \
    --resolution 256 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'a pixel art character with square red glasses, a baseball-shaped head and a orange-colored body on a dark background' \
        'a pixel art character with square orange glasses, a lips-shaped head and a red-colored body on a light background' \
        'a pixel art character with square blue glasses, a microwave-shaped head and a purple-colored body on a sunny background' \
        'a pixel art character with square red glasses, a baseball-shaped head and a blue-colored body on an orange background' \
        'a pixel art character with square red glasses' \
        'a pixel art character' \
        'square red glasses on a pixel art character' \
        'square red glasses on a pixel art character with a baseball-shaped head' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + 8 bit adam

Note that this training config keeps the batch size low and the learning rate high to get results fast with low resources. However, due to 8 bit adam, it will diverge eventually. If you want to train for longer, you will have to up the batch size and lower the learning rate.

Batch size: 16, Learning rate: 2e-5, Gives decent results in ~750 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    16        |          1                   |     16             |      20.1 GB       |
|    8        |          2                   |      16           |      15.6 GB       |
|    1        |          16                   |     16            |      10.7 GB       |

```sh
# TODO - update model path
accelerate launch training/training.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 2e-5 \
    --use_8bit_adam \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline-256-finetuned \
    --instance_data_dataset  'm1guelpf/nouns' \
    --image_key image \
    --prompt_key text \
    --resolution 256 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'a pixel art character with square red glasses, a baseball-shaped head and a orange-colored body on a dark background' \
        'a pixel art character with square orange glasses, a lips-shaped head and a red-colored body on a light background' \
        'a pixel art character with square blue glasses, a microwave-shaped head and a purple-colored body on a sunny background' \
        'a pixel art character with square red glasses, a baseball-shaped head and a blue-colored body on an orange background' \
        'a pixel art character with square red glasses' \
        'a pixel art character' \
        'square red glasses on a pixel art character' \
        'square red glasses on a pixel art character with a baseball-shaped head' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + lora

Batch size: 16, Learning rate: 8e-4, Gives decent results in 1000-1250 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    16        |          1                   |     16             |      14.1 GB       |
|    8        |          2                   |      16           |      10.1 GB       |
|    1        |          16                   |     16            |      6.5 GB       |

```sh
# TODO - update model path
accelerate launch training/training.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 8e-4 \
    --use_lora \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline-256-finetuned \
    --instance_data_dataset  'm1guelpf/nouns' \
    --image_key image \
    --prompt_key text \
    --resolution 256 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'a pixel art character with square red glasses, a baseball-shaped head and a orange-colored body on a dark background' \
        'a pixel art character with square orange glasses, a lips-shaped head and a red-colored body on a light background' \
        'a pixel art character with square blue glasses, a microwave-shaped head and a purple-colored body on a sunny background' \
        'a pixel art character with square red glasses, a baseball-shaped head and a blue-colored body on an orange background' \
        'a pixel art character with square red glasses' \
        'a pixel art character' \
        'square red glasses on a pixel art character' \
        'square red glasses on a pixel art character with a baseball-shaped head' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

### Finetuning the 512 checkpoint

These examples finetune on this [minecraft](https://huggingface.co/monadical-labs/minecraft-preview) dataset.

Example results:

![minecraft1](./assets/minecraft1.png) ![minecraft2](./assets/minecraft2.png) ![minecraft3](./assets/minecraft3.png)

#### Full finetuning

Batch size: 8, Learning rate: 8e-5, Gives decent results in 500-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      24.2 GB       |
|    4        |          2                   |     8             |      19.7 GB       |
|    1        |          8                   |     8             |      16.99 GB       |

```sh
# TODO - update model path
accelerate launch training/training.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 8e-5 \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --instance_data_dataset  'monadical-labs/minecraft-preview' \
    --prompt_prefix 'minecraft ' \
    --image_key image \
    --prompt_key text \
    --resolution 512 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'minecraft Avatar' \
        'minecraft character' \
        'minecraft' \
        'minecraft president' \
        'minecraft pig' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + 8 bit adam

Batch size: 8, Learning rate: 5e-6, Gives decent results in 500-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      21.2 GB       |
|    4        |          2                   |     8             |      13.3 GB       |
|    1        |          8                   |     8             |      9.9 GB       |

```sh
# TODO - update model path
accelerate launch training/training.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 5e-6 \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --instance_data_dataset  'monadical-labs/minecraft-preview' \
    --prompt_prefix 'minecraft ' \
    --image_key image \
    --prompt_key text \
    --resolution 512 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'minecraft Avatar' \
        'minecraft character' \
        'minecraft' \
        'minecraft president' \
        'minecraft pig' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

#### Full finetuning + lora 

Batch size: 8, Learning rate: 1e-4, Gives decent results in 500-1000 steps

| Batch Size | Gradient Accumulation Steps | Effective Total Batch Size | Memory Used |
|------------|-----------------------------|------------------|-------------|
|    8        |          1                   |     8             |      12.7 GB       |
|    4        |          2                   |     8             |      9.0 GB       |
|    1        |          8                   |     8             |      5.6 GB       |

```sh
# TODO - update model path
accelerate launch training/training.py \
    --output_dir <output path> \
    --train_batch_size <batch size> \
    --gradient_accumulation_steps <gradient accumulation steps> \
    --learning_rate 1e-4 \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --instance_data_dataset  'monadical-labs/minecraft-preview' \
    --prompt_prefix 'minecraft ' \
    --image_key image \
    --prompt_key text \
    --resolution 512 \
    --mixed_precision fp16 \
    --lr_scheduler constant \
    --validation_prompts \
        'minecraft Avatar' \
        'minecraft character' \
        'minecraft' \
        'minecraft president' \
        'minecraft pig' \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --validation_steps 250 \
    --gradient_checkpointing
```

### Styledrop

[Styledrop](https://arxiv.org/abs/2306.00983) is an efficient finetuning method for learning a new style.
It has an optional first stage to generates additional training samples. The additional training samples can
augment a small number of initial images such you need as little as 1 initial image.

#### Step 1

You need a small initial dataset of the style you want to teach the model. We will start with a single image.

![example](./training/A%20woman%20working%20on%20a%20laptop%20in%20[V]%20style.jpg)

All prompts should be of the form "<description of subject> in <identifier for style> style". The training script
uses the convention that the name of the file is its prompt. The identifier for the style can be the spelled out
name of the style e.g. "in a watercolor style". It can also be symbolic e.g. "[V]". Just keep the identifier consistent.

```sh
accelerate launch ./training/training.py \
    --output_dir <where to save checkpoints> \
    --mixed_precision fp16 \
    --report_to wandb \
    --use_lora \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --train_batch_size 8 \
    --lr_scheduler constant \
    --learning_rate 0.00003 \
    --validation_prompts \
        'A chihuahua walking on the street in [V] style' \
        'A banana on the table in [V] style' \
        'A church on the street in [V] style' \
        'A tabby cat walking in the forest in [V] style' \
    --instance_data_image './training/A woman working on a laptop in [V] style.jpg' \
    --max_train_steps 1000 \
    --checkpointing_steps 500 \
    --validation_steps 100
```

Generate a number of images and manually select those you think are of good quality. Select as many 
as you want but less than 12 is sufficient. Move the selected images and the initial image to a 
separate folder. 

TODO - "good quality" is not entirely correct. Do a better explanation of what images we are looking
for

```sh
python training/generate_images.py \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --write_images_to <where to save images> \
    --load_transformer_lora_from <output_dir from step 1>
```

e.g.

TODO put images here

#### Step 2

Train the model on the selected images and the initial images.

```sh
accelerate launch training.py \
    --output_dir <same output_dir as step 1> \
    --instance_data_dir <the directory you moved the good images to> \
    --resume_from latest \
    --mixed_precision fp16 \
    --report_to wandb \
    --use_lora \
    --pretrained_model_name_or_path openMUSE/diffusers-pipeline \
    --train_batch_size 8 \
    --lr_scheduler constant \
    --learning_rate 0.00003 \
    --validation_prompts \
        'A chihuahua walking on the street in [V] style' \
        'A chihuahua walking on the street in [V] style' \
        'A banana on the table in [V] style' \
        'A church on the street in [V] style' \
        'A tabby cat walking in the forest in [V] style' \
    --max_train_steps 1000 \
    --checkpointing_steps 500 \
    --validation_steps 100
```

## 4. Acknowledgements

## 5. Citation
```
@misc{patil-etal-2023-amused,
  author = {Suraj Patil and William Berman and Patrick von Platen},
  title = {Amused: An open MUSE model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/amused}}
}
```
