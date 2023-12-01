import argparse
import logging
from diffusers import AmusedPipeline
import os
from peft import PeftModel

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument("--style_descriptor", type=str, default="[V]")
    parser.add_argument(
        "--load_transformer_from",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--write_images_to", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    prompts = [
        f"A chihuahua in {args.style_descriptor} style",
        f"A tabby cat in {args.style_descriptor} style",
        f"A portrait of chihuahua in {args.style_descriptor} style",
        f"An apple on the table in {args.style_descriptor} style",
        f"A banana on the table in {args.style_descriptor} style",
        f"A church on the street in {args.style_descriptor} style",
        f"A church in the mountain in {args.style_descriptor} style",
        f"A church in the field in {args.style_descriptor} style",
        f"A church on the beach in {args.style_descriptor} style",
        f"A chihuahua walking on the street in {args.style_descriptor} style",
        f"A tabby cat walking on the street in {args.style_descriptor} style",
        f"A portrait of tabby cat in {args.style_descriptor} style",
        f"An apple on the dish in {args.style_descriptor} style",
        f"A banana on the dish in {args.style_descriptor} style",
        f"A human walking on the street in {args.style_descriptor} style",
        f"A temple on the street in {args.style_descriptor} style",
        f"A temple in the mountain in {args.style_descriptor} style",
        f"A temple in the field in {args.style_descriptor} style",
        f"A temple on the beach in {args.style_descriptor} style",
        f"A chihuahua walking in the forest in {args.style_descriptor} style",
        f"A tabby cat walking in the forest in {args.style_descriptor} style",
        f"A portrait of human face in {args.style_descriptor} style",
        f"An apple on the ground in {args.style_descriptor} style",
        f"A banana on the ground in {args.style_descriptor} style",
        f"A human walking in the forest in {args.style_descriptor} style",
        f"A cabin on the street in {args.style_descriptor} style",
        f"A cabin in the mountain in {args.style_descriptor} style",
        f"A cabin in the field in {args.style_descriptor} style",
        f"A cabin on the beach in {args.style_descriptor} style"
    ]

    logger.warning(f"generating image for {prompts}")

    logger.warning(f"loading models")

    pipe = AmusedPipeline.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant)

    if args.load_transformer_from is not None:
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer, os.path.join(args.load_transformer_from), is_trainable=False
        )

    pipe.to(args.device)

    logger.warning(f"generating images")

    os.makedirs(args.write_images_to, exist_ok=True)

    for prompt_idx in range(0, len(prompts), args.batch_size):
        images = pipe(prompts[prompt_idx:prompt_idx+args.batch_size]).images

        for image_idx, image in enumerate(images):
            prompt = prompts[prompt_idx+image_idx]
            image.save(os.path.join(args.write_images_to, prompt + ".png"))

if __name__ == "__main__":
    main(parse_args())