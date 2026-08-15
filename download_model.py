from transformers import AutoModelForCausalLM
import inspect
model = AutoModelForCausalLM.from_pretrained('microsoft/bitnet-b1.58-2B-4T', trust_remote_code=True)
print(model.model.layers[0].mlp)
print("="*40)
print(inspect.getsource(model.model.layers[0].mlp.forward))
print("="*40)
print(inspect.getsource(model.model.layers[0].self_attn.forward))
