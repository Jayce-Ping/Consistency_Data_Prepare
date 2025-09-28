import os
import re
import json
from socket import timeout
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
from itertools import combinations
from collections import defaultdict
import math
import time
from unittest import result
import argparse

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
import open_clip
from PIL import Image
from tqdm import tqdm

from consistency_scorer import ConsistencyScorer
from subfig_clipT import SubfigClipTScorer
from layout_scorer import GridLayoutScorer

# VLLM log filter
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


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

def eval(prompt_metadata_file, image_dir, result_file):
    with open(prompt_metadata_file) as f:
        metadatas = json.load(f)
    
    # model = 'GLM-4.1V-9B-Thinking'
    # model = "Qwen2.5-VL-7B-Instruct"
    model = 'InternVL3_5-8B'
    thinking = False
    consistency_scorer = ConsistencyScorer(
        model=model,
        criteria_path='data/prompt_consistency_criterion.json',
        thinking=thinking,
    )
    subfigclip_scorer = SubfigClipTScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
    layout_scorer = GridLayoutScorer(
        model=model,
        thinking=thinking,
    )
    # Load existing results
    if os.path.exists(result_file):
        with open(result_file) as f:
            existing_results = [json.loads(line) for line in f]

        existing_indices = {res['idx'] for res in existing_results}
        metadatas = [m for m in metadatas if m['idx'] not in existing_indices]
        print(f"Resuming from existing results, {len(existing_indices)} entries found, {len(metadatas)} remaining to process.")

    batch_size = 2

    batches = []
    for i, metadata in enumerate(metadatas):
        if i % batch_size == 0:
            batches.append([])

        idx = metadata['idx']
        prompt = metadata['prompt']
        image_path = os.path.join(image_dir, f"{idx}.png")
        batches[-1].append((idx, prompt, image_path, metadata))
    
    all_scores = defaultdict(list)
    for batch in tqdm(batches):
        indices, prompts, image_paths, metadatas = zip(*batch)
        images = [Image.open(path) for path in image_paths]
        batch_consistency_scores = asyncio.run(consistency_scorer(images, prompts, metadatas))
        batch_subfigclip_scores = subfigclip_scorer(images, prompts, metadatas).tolist()
        # Use all zero as placeholder
        # batch_subfigclip_scores = [0.0] * len(images)
        batch_layout_scores = asyncio.run(layout_scorer(images, prompts, metadatas))
        for idx, consistency_scores, subfigclip_scores, layout_scores in zip(indices, batch_consistency_scores, batch_subfigclip_scores, batch_layout_scores):
            all_scores['consistency_scores'].append(consistency_scores)
            all_scores['subfigclip_scores'].append(subfigclip_scores)
            all_scores['layout_scores'].append(layout_scores)

            with open(result_file, 'a') as f:
                f.write(json.dumps({
                    'idx': idx,
                    'consistency_scores': consistency_scores,
                    'subfigclip_scores': subfigclip_scores,
                    'layout_scores': layout_scores
                }))
                f.write('\n')
    
    all_scores = {k: np.array(v) for k, v in all_scores.items()}
    for k, v in all_scores.items():
        print(k, " has mean = ", np.mean(v), " std = ", np.std(v))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate consistency of generated images.")
    parser.add_argument('--prompt_file', type=str, default='data/prompt.json', help='Path to the prompt metadata JSON file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing generated images.')
    parser.add_argument('--result_file', type=str, required=True, help='File to save evaluation results (jsonl).')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    prompt_file = args.prompt_file
    image_dir = args.image_dir
    result_file = args.result_file

    result_dir = os.path.dirname(result_file)
    os.makedirs(result_dir, exist_ok=True)
    
    eval(prompt_file, image_dir, result_file)

if __name__ == '__main__':
    main()

