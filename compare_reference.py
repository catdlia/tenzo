import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'microsoft/bitnet-b1.58-2B-4T'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "What is AI?"
inputs = tokenizer(prompt, return_tensors='pt')
input_ids = inputs['input_ids']

print(f"Prompt: '{prompt}'")
print("HuggingFace Token IDs:", input_ids[0].tolist())
print("Decoded Tokens:", [tokenizer.decode([t]) for t in input_ids[0].tolist()])

with torch.no_grad():
    # 1. Full 30-layer model forward
    out_full = model(**inputs)
    logits_full = out_full.logits[0, -1, :]
    top5_full = torch.topk(logits_full, 5)
    print("\n--- Full 30-Layer Model Top-5 Predictions ---")
    for val, id_ in zip(top5_full.values, top5_full.indices):
        print(f"Token ID {id_.item():6d} ({repr(tokenizer.decode([id_.item()]))}): logit={val.item():.4f}")

    # 2. Slice first 2 layers only to match our 2-layer test build
    model_2layer = AutoModelForCausalLM.from_pretrained(model_name)
    model_2layer.model.layers = model_2layer.model.layers[:2]
    
    out_2layer = model_2layer(**inputs)
    logits_2layer = out_2layer.logits[0, -1, :]
    top5_2layer = torch.topk(logits_2layer, 5)
    print("\n--- 2-Layer Test Model Top-5 Predictions ---")
    for val, id_ in zip(top5_2layer.values, top5_2layer.indices):
        print(f"Token ID {id_.item():6d} ({repr(tokenizer.decode([id_.item()]))}): logit={val.item():.4f}")

