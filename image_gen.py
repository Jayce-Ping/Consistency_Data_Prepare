import os
import json
import argparse
from tqdm import tqdm
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from peft import PeftModel

def load_json(json_file : str):
    with open(json_file, 'r') as f:
        return json.load(f)


def load_flux_pipeline(model_path, **load_kwargs):
    pipeline = FluxPipeline.from_pretrained(model_path, **load_kwargs)
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    return pipeline

def load_sd3_pipeline(model_path, **load_kwargs):
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path, **load_kwargs)
    pipeline.transformer.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)

    return pipeline

def image_gen(prompt_metadata_file : str, model_path : str, model_type = None, output_dir : str = 'output', lora_path: str = None, num_images_per_prompt: int = 1):
    load_kwargs = {
        'device_map': 'balanced',
        'torch_dtype': torch.bfloat16
    }
    if model_type is None:
        if 'flux' in model_path.lower():
            model_type = 'flux'
        elif 'stable' in model_path.lower():
            model_type = 'sd'
        else:
            model_type = None

    if model_type == 'flux':
        pipeline = load_flux_pipeline(model_path, **load_kwargs)
    elif model_type == 'sd':
        pipeline = load_sd3_pipeline(model_path, **load_kwargs)
    else:
        raise NotImplementedError("Please include `flux` or `stable` in the model_path to specify the model type")

    if lora_path:
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_path)
        # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
        pipeline.transformer.set_adapter("default")

    os.makedirs(output_dir, exist_ok=True)
    inference_steps = 20 if model_type == 'flux' else 40
    batch_size = 4
    file_ext = os.path.splitext(prompt_metadata_file)[1]
    if file_ext == '.json':
        prompt_metadata = load_json(prompt_metadata_file)
    elif file_ext == '.jsonl':
        prompt_metadata = [json.loads(line) for line in open(prompt_metadata_file, 'r')]
    else:
        raise ValueError("Unsupported file format. Please use .json or .jsonl")

    # Since each image may have different height/width, we cannot directly use batch inference here.
    for metadata in tqdm(prompt_metadata):
        idx = metadata['idx']
        prompt = metadata['prompt']
        h = metadata['height']
        w = metadata['width']
        max_sequence_length = 512
        seed = metadata.get('seed', None)
        output_image_path = os.path.join(output_dir, f"{idx}.png")

        output = pipeline(
            prompt=prompt,
            guidance_scale=3.5,
            height=h,
            width=w,
            num_inference_steps=inference_steps,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt
        )
        for img_id, image in enumerate(output.images):
            image.save(os.path.join(output_dir, f"{idx}_{img_id}.png"))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images")
    parser.add_argument('--model', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_type', default=None)
    parser.add_argument('--prompt', default='data/prompt.json')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--lora_path', default=None, help='Path to LoRA folder')
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    image_gen(args.prompt, args.model, args.model_type, args.output_dir, args.lora_path, args.num_images_per_prompt)

if __name__ == '__main__':
    main()
