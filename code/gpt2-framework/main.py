from transformers import AutoTokenizer, AutoModelForCausalLM
from greedy import greedy_search
from beam import beam_search
from sampling import sampling_search


# gpt2 = "gpt2"
dialogpt = "microsoft/DialoGPT-medium"

# change the model here
tokenizer = AutoTokenizer.from_pretrained(dialogpt)
model = AutoModelForCausalLM.from_pretrained(dialogpt)

# print("-----Greedy search-----")
# greedy_search(model, tokenizer)
# print("-----Beam search-----")
# beam_search(model, tokenizer)
print("-----Sampling-----")
sampling_search(model, tokenizer)