import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
DB_START_TOKEN = "<|db_entity|>"          # Begins a lookup call
DB_SEP_TOKEN = "<|db_relationship|>"                # Separates entity and relation in the query
DB_RETRIEVE_TOKEN = "<|db_return|>"   # Signals insertion point for returned value
DB_END_TOKEN = "<|db_end|>"


def normalize_db_format(text):
        text = re.sub(r'<\|db_entity\|>\s*', DB_START_TOKEN, text)
        text = re.sub(r'<\|db_relationship\|>\s*', DB_SEP_TOKEN, text)
        text = re.sub(r'<\|db_return\|>\s*', DB_RETRIEVE_TOKEN, text)
        text = re.sub(r'<\|db_end\|>\s*', DB_END_TOKEN, text)
        return text

def _decode_with_special_tokens(outputs, tokenizer, input_len, input_text):
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        output_text = normalize_db_format(output_text)

        if input_text in output_text:
            output_text = output_text.split(input_text)[-1]
        else:
            output_text = tokenizer.decode(outputs[0][input_len:], clean_up_tokenization_spaces=True) 
            output_text = normalize_db_format(output_text)
            # logger.info(f"decode again: {output_text}")
        return output_text  

def add_dblookup_special_tokens(tokenizer, config=None):
    """
    Add special dblookup tokens to tokenizer and optionally update config.
    """
    db_tokens = {
        "entity": DB_START_TOKEN,
        "relationship": DB_SEP_TOKEN,
        "return": DB_RETRIEVE_TOKEN,
        "end": DB_END_TOKEN,
    }

    new_tokens = list(db_tokens.values())
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {num_added} tokens to the vocabulary")

    if config:
        config.vocab_size += num_added
        print(f"Updated vocab_size to {config.vocab_size}")

    return tokenizer, config


# tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
# tok , _= add_dblookup_special_tokens(tok)
model_path = "/home/rtn27/LMLM_develop/training/qwen3-1.7b/checkpoints/_full_ep10_bsz32_new_qa"
tok = AutoTokenizer.from_pretrained(model_path, use_fast = True)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

text = "<thinking> First, we need to determine which college Marjorie Hass is the 20th president of formerly affiliated with. Marjorie Hass's role as 20th president of <|db_entity|>Marjorie Hass<|db_relationship|>"
inputs = {'input_ids': torch.tensor([[ 14582,    510,   3838,   8817,    572,    279,   7770,   2876,     73,
          29203,  42022,    374,    279,    220,     17,     15,    339,   4767,
            315,  33733,  36406,    448,   5267,  16141,    510,  13708,  15736,
             29,   5512,     11,    582,   1184,    311,   8253,    892,   7770,
           2876,     73,  29203,  42022,    374,    279,    220,     17,     15,
            339,   4767,    315,  33733,  36406,    448,     13,   2876,     73,
          29203,  42022,    594,   3476,    438,    220,     17,     15,    339,
           4767,    315,    220, 151669,  12061,     73,  29203,  42022, 151670]],
       device='cuda:0'), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}

text = tok.decode(inputs["input_ids"][0], return_tensors = "pt")
print("text")
input_len = inputs["input_ids"].shape[1]
with torch.no_grad():
    outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens = 1,
                    do_sample = False
                )
    print("inputs:\n\n ", inputs, "\n\n")
    print("outputs:\n\n ", outputs, "\n\n")
res = _decode_with_special_tokens(outputs, tok, input_len, text)
print("---",res,"---", "\n\n")

prompt = "hello" + " "
res = tok(prompt, return_tensors = "pt").to("cuda")
print(res, "\n\n")


