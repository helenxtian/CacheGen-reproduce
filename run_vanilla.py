
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
import json
from src.utils import *
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--bins", type=int)
p.add_argument("--results_dir", type=str, default = "results")
p.add_argument("--results_str", type=str, default = "gt")
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", type=int)
p.add_argument("--low_mem_prefill", action="store_true",
               help="prefill all but the last prompt token without logits, then generate "
                    "from that cache (needed for 20K+ token prompts on 16 GB GPUs)")
args = p.parse_args()
if __name__ == "__main__":
    # Check if save_dir exists
    
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    data =  load_testcases(dataset_path(args.dataset_name))
    layer_to_device_id = {}
    average_acc = []
    average_size = []
    for doc_id in range(args.start, args.end):
        # Reference size: the uncompressed fp16 KV cache saved by main.py
        raw_path = f"{args.save_dir}/raw_kv_{doc_id}.pt"
        if os.path.exists(raw_path):
            average_size += [os.path.getsize(raw_path)/1e6]
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        if args.low_mem_prefill:
            past = prefill_kv(model, input_ids[:, :-1])
            generated = model.generate(input_ids, past_key_values=past, max_new_tokens = 20)
        else:
            generated = model.generate(input_ids,  max_new_tokens = 20)
        prediction = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"doc id: {doc_id}", tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))
        if args.calculate_metric == 1:
            if base_dataset(args.dataset_name) == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                average_acc += [metric]
            elif base_dataset(args.dataset_name) in ("nqa", "tqa"):
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                average_acc += [metric]
    if args.calculate_metric == 1:
        if base_dataset(args.dataset_name) == "longchat":
            print("Average vanilla accuracy is: ", np.mean(average_acc))
        elif base_dataset(args.dataset_name) in ("nqa", "tqa"):
            print("Average vanilla F1 score is: ", np.mean(average_acc))
    if average_size:
        print("Average fp16 KV cache size is: ", np.mean(average_size))
    save_results(args.results_dir, args.dataset_name, "vanilla", "fp16",
                 average_size, average_acc)
