import argparse
import os
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import sys


from datasets import load_dataset, concatenate_datasets
import accelerate
from contextlib import contextmanager
@contextmanager
def null_context(*args, **kwargs):
    yield
accelerate.init_empty_weights = null_context

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from PIL import Image
import math
def batch_to_pil_and_concat(video, mode='horizontal'):
    if video.dtype != np.uint8:
        video = video.astype(np.uint8)
    pil_images = [Image.fromarray(frame, 'RGB') for frame in video]
    
    if mode == 'horizontal':
        total_width = sum(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in pil_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
    elif mode == 'vertical':
        total_height = sum(img.height for img in pil_images)
        max_width = max(img.width for img in pil_images)
        combined = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for img in pil_images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError("Unsupported mode. Use 'horizontal' or 'vertical'.")
    
    return combined

def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
        # fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(lst / n)  # integer division
    return [[i,i+chunk_size-1] for i in range(0, lst, chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process(line, args, tokenizer, image_processor, model_config):
    qs = line["question"]
    if line['options']:
        qs += f"{line['options']}"
    # print(qs)
    # assert 0
        qs += f"\n{args.question_extension}"
    else:
        qs += f"\n{args.question_extension_2}"

    if line["scene_name"] is not None:
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if line["scene_name"] is None:
        image = None
        image_size = None
        image_tensor = None
    else:
        image_path = f"{args.data_root}/{line['dataset']}/{line['scene_name']}.mp4"
        print(image_path)

        video = load_video(image_path, args.max_frames_num)
        video = batch_to_pil_and_concat(video, mode='horizontal')
        video_size = [video.size]
        video = process_images([video], image_processor, model_config)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    return input_ids, video_size,video, prompt


def eval_model(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model
    # disable_torch_init()  # DO NOT ENABLE THIS: KILLS PERFORMANCE
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = load_dataset("nyu-visionx/VSI-Bench", split="test")

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    basename = os.path.basename(answers_file)
    basename = os.path.splitext(basename)[0]
    answers_dir = os.path.dirname(answers_file)
    chunk_fname = f"{basename}_{args.chunk_idx}.jsonl"
    chunk_file = os.path.join(answers_dir, chunk_fname)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)

    ans_file = open(chunk_file, "w")

    idx = -1
    valid_chunk = get_chunk(len(questions), args.num_chunks, args.chunk_idx)
    print(valid_chunk)
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        if idx<valid_chunk[0] or idx>valid_chunk[1]:
            continue
    
        input_ids, video_sizes, image_tensor, prompt = process(line, args, tokenizer, image_processor, model.config)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                image_sizes=video_sizes,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "questionId": idx,
            "scene_name": line["scene_name"],
            "prompt": prompt,
            "answer": outputs,
            "gt_answer": line["ground_truth"],
            "category": line["question_type"],
            "dataset": line["dataset"], 
            "frame": args.max_frames_num, 
            "model_id": model_name
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="nyu-visionx/cambrian-8b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="./answers/answers.jsonl")
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--question_extension_2", type=str, default="Please answer the question using a single word or phrase.")
    parser.add_argument("--data_root", type=str, default="your save address")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_frames_num", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_model(args)
