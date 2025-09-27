import os
import re
import json
from typing import List, Tuple, Union
from io import BytesIO
import base64
import logging
import asyncio
import time
from unittest import result

import torch
import numpy as np
import openai
from openai import OpenAI, AsyncOpenAI
from PIL import Image


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


def get_yes_prob_from_completion(completion : openai.ChatCompletion) -> float:
    if completion is None:
        return 0.0

    logprobs = completion.choices[0].logprobs
    if logprobs:
        # Use logprobs to compute, score = P('yes') / (P('yes') + P('no'))
        # score = 1 / (1 + exp(logprob('no') -  logprob('yes')))
        # Same formular for logits as well. Since the sum term will cancel out.
        # Use uppercase only here.
        token_logprobs = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
        yes_logprob = token_logprobs.get('Yes', float('-inf'))
        no_logprob = token_logprobs.get('No', float('-inf'))

        if yes_logprob == float('-inf') and no_logprob == float('-inf'):
            # When inf - inf encountered, give 0.0 score.
            score = 0.0 # 0.0
        else:
            diff = torch.tensor(yes_logprob - no_logprob, dtype=torch.float64)
            score = torch.sigmoid(diff).item()
    else:
        # log_prob cannot be derived here. How to calculate?
        # TODO
        score = 0.0

    return score



class GridLayoutScorer:
    def __init__(
            self,
            api_key: str = 'dummy-key',
            base_url: str = 'http://127.0.0.1:8000/v1',
            model='Qwen2.5-VL-7B-Instruct',
            max_concurrent=12,  # 2x2 grid has 6 pair of images to compare. 12 for at most 2 batches at once.
            max_retries=10,
            timeout=60,
            thinking=False,
        ):
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.thinking = thinking

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    @torch.no_grad()
    async def __call__(self, images : list[Image.Image], prompts : list[str], metadatas : list[dict]) -> list[float]:
        assert len(prompts) == len(images), "Length of prompts and images must match"

        # Create a global semaphore for overall concurrency control
        global_semaphore = asyncio.Semaphore(self.max_concurrent)

        # Process all images concurrently
        async def process_single_image(prompt, image, metadata):
            async with global_semaphore:
                return await self.compute_layout_score(prompt, image, metadata)

        # Process all images concurrently
        tasks = [
            process_single_image(prompt, image, metadata) 
            for prompt, image, metadata in zip(prompts, images, metadatas)
        ]

        final_scores = await asyncio.gather(*tasks)
        return final_scores
    
    async def compute_layout_score(
            self,
            prompt : str,
            image : Image.Image,
            metadata : dict,
            top_logprobs: int = 20,
            threshold = 0.9,
        ) -> float:
        grid_info = extract_grid_info(prompt)
        messages = [
            {
                "role": "user",
                "content":
                [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": f"Is it a {grid_info} grid layout image? Please answer Yes or No."},
                ]
            }
        ]

        for attempt in range(self.max_retries):
            try:
                if not self.thinking:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0, # Deterministic result, no use for logprobs, actually.
                        max_completion_tokens=1,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout
                    )
                    break
                else:
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        logprobs=True,
                        top_logprobs=top_logprobs,
                        timeout=self.timeout,
                        stream=True
                    )
                    async for chunk in stream:
                        # Skip thinking process to reach the first token after <answer>
                        if chunk.choices[0].delta.content.strip() == '<answer>':
                            completion = await anext(stream, None)
                            if completion and completion.choices[0].delta.content == '<|begin_of_box|>':
                                completion = await anext(stream, None)
                            break
                        else:
                            completion = None
                    break
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    completion = None

        if completion is None:
            return 0.0
        else:
            choice = completion.choices[0]
            if hasattr(choice, 'message'):
                content = choice.message.content.strip().lower()
            elif hasattr(choice, 'delta'):
                content = choice.delta.content.strip().lower()

            if 'yes' in content:
                return 1.0
            else:
                return 0.0
            yes_prob = get_yes_prob_from_completion(completion)
            if yes_prob > threshold:
                return 1
            else:
                return 0
