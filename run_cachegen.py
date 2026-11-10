
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import os
import time
import pickle
import torch
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
import json
from src.utils import *
from src.delta_codec import DeltaCacheGenCodec

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--encoded_dir", type=str, default = None)
p.add_argument("--results_dir", type=str, default = None)
p.add_argument("--results_str", type=str, default = "results")
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", type=int)
# Reproduction change: the original code hard-coded QUANT_LEVEL=2, which gives a
# single point. Fig. 9 needs several points on the size/quality curve.
p.add_argument("--quant_level", type=str, default="2", choices=["1", "2", "3"],
               help="CacheGen encoding level (1 = smallest, 3 = highest quality)")
# Reproduction addition: the paper's token-wise delta encoding (Sec. 5.2), which the
# released codec omits. See src/delta_codec.py.
p.add_argument("--codec", type=str, default="cachegen", choices=["cachegen", "delta"],
               help="cachegen = released codec; delta = our reimplementation with delta encoding")
p.add_argument("--anchor_mode", type=str, default="level", choices=["level", "int8"],
               help="delta codec only: how anchor tokens are stored (see src/delta_codec.py)")
p.add_argument("--group_size", type=int, default=10, help="delta codec only: tokens per group")

args = p.parse_args()


if __name__ == "__main__":
    os.makedirs(args.encoded_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    data =  load_testcases(DATASET_TO_PATH[args.dataset_name])
    # Both the encoder and decoder read this when they are constructed
    os.environ["QUANT_LEVEL"] = args.quant_level
    if args.codec == "delta":
        method = "cachegen_delta"
        setting = f"level{args.quant_level}" + ("_int8anchor" if args.anchor_mode == "int8" else "")
    else:
        method, setting = "cachegen", f"level{args.quant_level}"
    encoded_dir = os.path.join(args.encoded_dir, method, setting)
    os.makedirs(encoded_dir, exist_ok=True)

    layer_to_device_id = {}
    kv = pickle.load(open(f"{args.save_dir}/raw_kv_{args.start}.pkl", "rb"))
    for i in range(len(kv)):
        layer_to_device_id[i] = kv[i][0].device.index
    del kv

    # Start encoding
    kv_tokens = {}
    avg_size = []
    breakdowns = []
    if args.codec == "delta":
        delta_codec = DeltaCacheGenCodec(args.model_id, group_size=args.group_size,
                                         anchor_mode=args.anchor_mode)
    for doc_id in range(args.start, args.end):
        key_value = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        if args.codec == "delta":
            bytes = delta_codec.to_bytes(key_value)
            breakdowns.append(delta_codec.last_stats)
            print("  size breakdown:", {k: round(v, 4) for k, v in delta_codec.last_stats.items()})
        else:
            lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
            meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
            cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
            bytes = cachegen_serializer.to_bytes(key_value)
        pickle.dump(bytes, open(f"{encoded_dir}/{doc_id}.pkl", "wb"))
        # Keyed by doc_id (the original indexed a list by doc_id, which broke for --start != 0)
        kv_tokens[doc_id] = key_value.shape[-2]
        avg_size += [len(bytes)/1e6]
        print(f"doc {doc_id}: {kv_tokens[doc_id]} tokens -> {len(bytes)/1e6:.1f} MB")
        del key_value
        torch.cuda.empty_cache()

    # Start inferencing. Decode one document at a time, right before using it,
    # instead of keeping all decoded KV caches in CPU RAM (about 1.3 GB each).
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    average_acc = []
    for doc_id in range(args.start, args.end):
        os.environ['DOC_ID'] = str(doc_id)
        print("Running inference for doc_id: ", doc_id)
        bytes = pickle.load(open(f"{encoded_dir}/{doc_id}.pkl", "rb"))
        if args.codec == "delta":
            decoded_kv = delta_codec.from_bytes(bytes).cuda()
        else:
            lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_tokens[doc_id])
            meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
            deserializer = CacheGenDeserializer(lmcache_config, meta_data)
            decoded_kv = deserializer.from_bytes(bytes).cuda()
        decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        if args.calculate_metric == 1:
            if args.dataset_name == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                average_acc += [metric]
            elif args.dataset_name == "nqa" or args.dataset_name == "tqa":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                average_acc += [metric]
        if args.dataset_name == "longchat":
            print(prediction, data[doc_id]['label'][0])
        del decoded_kv, output
        torch.cuda.empty_cache()
    if args.dataset_name == "longchat":
        metric_name = "accuracy"
    else:
        metric_name = "F1 score"
    if args.calculate_metric == 1:
        print(f"Average {method} ({setting}) {metric_name} is: ", np.mean(average_acc))
    print(f"Average size of KV cache: {np.mean(avg_size)}MB")
    extra = {"avg_tokens": float(np.mean(list(kv_tokens.values())))}
    if breakdowns:
        extra["avg_breakdown"] = {k: float(np.mean([b[k] for b in breakdowns])) for k in breakdowns[0]}
        extra["group_size"] = args.group_size
        extra["anchor_mode"] = args.anchor_mode
    save_results(args.results_dir, args.dataset_name, method, setting, avg_size, average_acc, extra=extra)
