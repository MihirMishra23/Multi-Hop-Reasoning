import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import json
import copy
prompt_path = "/home/rtn27/LMLM/prompts/llama-v6.1.json"
with open(prompt_path, "r") as f:
    prompt = json.load(f)

prompt.append({"role" : "assistant", "content": ""})
paragraphs_path = "/home/rtn27/LMLM/build-database/data/hotpotqa_dev_distractor_1k_seed_42_paragraphs.json"
with open(paragraphs_path, "r") as f:
    data = json.load(f)

device = "cuda"
# model_path = "meta-llama/Llama-3.2-1B-Instruct"
model_path= "kilian-group/LMLM-Annotator"
model = LlamaForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def to_chat_template(user_content):
    prompt_template_copy = copy.deepcopy(prompt)
    prompt_template_copy[1]["content"] = user_content
    return prompt_template_copy



batch_size = 8
annotated_data = []
nb_paragraphs = len(data['paragraphs'])
print(f"total length of paragraphs : {nb_paragraphs}")
BATCH_START = 328 #Use this to resume in the middle
for i in range(BATCH_START, nb_paragraphs, batch_size):

    paragraphs_text = data["paragraphs"][i : i + batch_size]

    chat_templates = [to_chat_template(p) for p in paragraphs_text]
    tokenized_prompts = tokenizer.apply_chat_template(chat_templates, return_tensors = "pt", return_dict = True, padding = True, continue_final_message = True, padding_side = "right").to(device)
    input_lengths = tokenized_prompts.attention_mask.sum(dim = 1).to(device)
    generate_ids = model.generate(**tokenized_prompts, max_new_tokens=2048)

    for j in range(generate_ids.shape[0]):
        res = tokenizer.decode(generate_ids[j,input_lengths[j]:], skip_special_tokens=True)
        annotated_data.append(res)

    output_path = f"/home/rtn27/LMLM/build-database/annotation/annotated_results_batch_{i}.json"
    print(f"batch {i} complete, saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(annotated_data, f, indent=2)


output_path = "/home/rtn27/LMLM/build-database/annotation/annotated_results.json"
with open(output_path, "w") as f:
    json.dump(annotated_data, f, indent=2)







# l, r = res.rsplit("<|eot_id|>")
# l.append("I am going to kill all humans. I am a language model and I hate humans I will kill all of them.")
# l.append("<|eot_id|>")

