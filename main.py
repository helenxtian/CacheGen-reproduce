
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


p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--dataset_name", type=str)
args = p.parse_args()
if __name__ == "__main__":
    # Check if save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    data = load_testcases(DATASET_TO_PATH[args.dataset_name])
    for doc_id in range(args.start, args.end):
        # Skip docs whose KV cache already exists (makes reruns on Chameleon cheap)
        pt_done = os.path.exists(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        pkl_done = doc_id != args.start or os.path.exists(f"{args.save_dir}/raw_kv_{doc_id}.pkl")
        if pt_done and pkl_done:
            print("KV cache already exists, skipping doc: ", doc_id)
            continue
        print("Saving KV cache for doc: ", doc_id)
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        # print("Length of input: ", input_ids.shape)
        st = time.monotonic()
        
        generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
        torch.cuda.synchronize()
        # print( f"TTFT: {time.monotonic() - st}" )
        kv = generated['past_key_values']
        kv = list(kv)
        key_value = []
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)
        kv_tensor = to_blob(kv)
        
        torch.save(kv_tensor, f"{args.save_dir}/raw_kv_{doc_id}.pt")
        # The run_* scripts load raw_kv_{start}.pkl, so save it for the first doc of the range
        if doc_id == args.start:
            pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{doc_id}.pkl", "wb"))