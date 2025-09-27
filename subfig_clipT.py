import os
import re
import json
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations
import time

import torch
import numpy as np
import open_clip
from PIL import Image

def pil_image_to_base64(image, format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_qwen

def divide_prompt(prompt):
    # seqis like ". [TOP-LEFT]:"
    match_sep = re.compile(r"\.\s+[A-Z0-9-\[\]]+:")
    seps = match_sep.findall(prompt)
    # Add '.' for each sentence
    sub_prompts = [
        p + '.' if p.strip()[-1] != '.' else p
        for p in re.split('|'.join(map(re.escape, seps)), prompt)
    ]
    return sub_prompts

def divide_image(image, grid_info : tuple[int, int]):
    assert len(grid_info) == 2, "grid_info must be a tuple of two integers (a, b)"

    a, b = grid_info
    width, height = image.size

    grid_cells = []
    cell_width = width // b
    cell_height = height // a

    # 2x2 grid
    # | 1 | 2 |
    # | 3 | 4 |
    # [
    # (0, 0, cell_width, cell_height),
    # (cell_width, 0, 2 * cell_width, cell_height),
    # (0, cell_height, cell_width, 2 * cell_height),
    # (cell_width, cell_height, 2 * cell_width, 2 * cell_height)
    # ]

    for i in range(a):
        for j in range(b):
            upper = i * cell_height
            left = j * cell_width
            right = left + cell_width
            lower = upper + cell_height
            grid_cells.append(image.crop((left, upper, right, lower)))

    return grid_cells

def extract_grid_info(prompt) -> tuple[int, int]:
    # Grid can be represented as int x int, or int ⨉ int. ⨉ has unicode \u2a09
    match = re.findall(r'(\d+)\s*[x⨉]\s*(\d+)', prompt)
    if len(match) == 0:
        return (1, 1)

    return (int(match[0][0]), int(match[0][1]))

class SubfigClipTScorer(torch.nn.Module):
    """
        Scorer for sub-images clip-T-score - align Image-Text semantics.
    """
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",         
            pretrained="laion2b_s32b_b79k",
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    @torch.no_grad()
    def __call__(self,
        images: List[Image.Image],
        prompts : List[str],
        metadata : List[dict]
    ) -> np.ndarray:
        """
            Compute the average CLIP score for each subfigure-subprompt pair in a batch of images and prompts.

            Args:
                images (List[Image.Image]): List of PIL images to be evaluated.
                prompts (List[str]): List of prompts corresponding to each image. Each prompt may contain subprompts for subfigures.
                metadata (List[dict]): List of metadata dictionaries for each image (not used in this function).

            Returns:
                np.ndarray: Array of average CLIP scores for each image, computed as the mean of the diagonal of the subfigure-subprompt CLIP score matrix.
        """
        scores = []
        for image, prompt in zip(images, prompts):
            grid_info = extract_grid_info(prompt)
            sub_images = divide_image(image, grid_info)
            sub_prompts = divide_prompt(prompt)[1:]

            clip_matrix = self.compute_ClipT_matrix(sub_prompts, sub_images)
            # clip_scores = clip_matrix.softmax(dim=-1).diagonal().numpy()
            clip_scores = clip_matrix.diagonal().numpy()

            scores.append(np.mean(clip_scores))
        
        return np.array(scores)

    @torch.no_grad()
    def compute_ClipT_matrix(self, text : Union[str, List[str]], image : Union[Image.Image, list[Image.Image]]):
        input_texts = [text] if isinstance(text, str) else text
        input_images = [image] if isinstance(image, Image.Image) else image
        
        text_tokens = self.tokenizer(input_texts).to(self.device)
        images = torch.stack([self.preprocess(img).to(self.device) for img in input_images], dim=0)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            clip_scores = (image_features @ text_features.T)
            return clip_scores.cpu()