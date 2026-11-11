
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import math
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
args = p.parse_args()
if __name__ == "__main__":
    # Check if save_dir exists
    
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    data =  load_testcases(dataset_path(args.dataset_name))
    layer_to_device_id = {}
    kv = pickle.load(open(f"{args.save_dir}/raw_kv_{args.start}.pkl", "rb"))
    for i in range(len(kv)):
        layer_to_device_id[i] = kv[i][0].device.index
    average_acc = []
    average_size = []
    for doc_id in range(args.start, args.end):
        raw_kv = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        kv, max_tensors = default_quantization(raw_kv, args.bins, layer_to_device_id)
        # Reproduction change: the original measured the size of the saved .pt files,
        # which are always int8, so 3-/4-bit runs reported the same size as 8-bit.
        # Count the actual bit-width instead: log2(bins) bits per element, plus the
        # per-token scaling factors (max_tensors) in their native dtype.
        bits = math.log2(args.bins)
        size_bytes = kv.numel() * bits / 8 + max_tensors.numel() * max_tensors.element_size()
        average_size += [size_bytes/1e6]

        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        # Dequantize in memory (no need to round-trip through disk)
        kv = dequantize_kv(kv, max_tensors, args, layer_to_device_id)
        
        generated = model.generate(input_ids, past_key_values=kv, max_new_tokens = 20)
        prediction = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"doc id: {doc_id}", tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))
        if args.calculate_metric == 1:
            if base_dataset(args.dataset_name) == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                average_acc += [metric]
            elif base_dataset(args.dataset_name) in ("nqa", "tqa"):
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                average_acc += [metric]
    if base_dataset(args.dataset_name) == "longchat":
        metric_name = "accuracy"
    else:
        metric_name = "F1 score"
    if args.calculate_metric == 1:
        print(f"Average quantization ({int(math.log2(args.bins))}-bit) {metric_name} is: ", np.mean(average_acc))
    print("Average size is: ", np.mean(average_size))
    save_results(args.results_dir, args.dataset_name, "quant", f"{int(math.log2(args.bins))}bit",
                 average_size, average_acc)