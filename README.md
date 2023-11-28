# amused

[[Paper]]()
[[Models]]()
[[Colab]]()
[[Training Code]]()

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
image.save(f'text2image_256.png')
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
image.save(f'text2image_512.png')
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
        "<TODO>"
    )
    .resize((256, 256))
    .convert("RGB")
)

image = pipe(prompt, input_image, strength=0.7, generator=torch.Generator('cuda').manual_seed(3)).images[0]
image.save(f'image2image_256.png')
```

![image2image_256_orig](./assets/image2image_256_orig.png) 
![image2image_256](./assets/image2image_256.png)

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
        "<TODO>"
    )
    .resize((512, 512))
    .convert("RGB")
)

image = pipe(prompt, input_image, generator=torch.Generator('cuda').manual_seed(15)).images[0]
image.save(f'image2image_512.png')
```

![image2image_512_orig](./assets/image2image_512_orig.png) 
![image2image_512](./assets/image2image_512.png)

### Inpainting

#### 256x256 model

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
        "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg"
    )
    .resize((512, 512))
    .convert("RGB")
)
mask = (
    load_image(
        "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
    )
    .resize((512, 512))
    .convert("L")
)
image = pipe(prompt, input_image, mask).images[0]
```

## 2. Performance

### torch.compile

### flash attention

## 3. Training

## 4. Finetuning

## 5. Acknowledgements

## 6. Citation
