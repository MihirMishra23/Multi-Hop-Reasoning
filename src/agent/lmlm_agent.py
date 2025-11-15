from src.agent.agent import Agent, AgentStep
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmlm.database.database_manager import DatabaseManager
from transformers import LogitsProcessor


DB_START_TOKEN = "<|db_entity|>"          # Begins a lookup call
DB_SEP_TOKEN = "<|db_relationship|>"                # Separates entity and relation in the query
DB_RETRIEVE_TOKEN = "<|db_return|>"   # Signals insertion point for returned value
DB_END_TOKEN = "<|db_end|>"

def create_prompt_from_query(query):
    return f"Question:\n{query}\nAnswer:\n"

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


class LogitBiasProcessor(LogitsProcessor):
    def __init__(self, bias_dict: dict):
        """
        bias_dict: {token_id: bias_value (positive = more likely)}
        """
        super().__init__()
        self.bias_dict = bias_dict

    def __call__(self, input_ids, scores):
        for token_id, bias in self.bias_dict.items():
            scores[:, token_id] += bias
        return scores

class LMLMAgent(Agent):
    def __init__(self, model_path = "/home/rtn27/LMLM_develop/training/llama3.2-1b/checkpoints/_full_ep10_bsz128_new_qa", database_path = "../LMLM/hotpotqa_annotation_results/extracted_database_lookups.json", similarity_threshold = 0.6):
        self.model_path = model_path
        self.database_path = database_path
        self.db = DatabaseManager()
        self.db.load_database(database_path)
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if self.device=="cuda" else None)
        self.model.to(self.device).eval()
        self.similarity_threshold = similarity_threshold

    def run(self, query : str,  max_tokens = 256, temperature = 0.0):
        count = 0
        prompt = create_prompt_from_query(query)
        while (count < max_tokens):
            count += 1
            inputs = self.tok(prompt, return_tensors = "pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            #### Step 2: Generate until DB_RETRIEVE_TOKEN
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens = 1,
                    do_sample = False if temperature == 0.0 else True
                )
        
            
            output_text = _decode_with_special_tokens(outputs, self.tok, input_len, prompt)

            prompt += output_text

            if prompt.endswith("<answer/>"):
                break

            # Check if <|db_return|> is present
            if DB_RETRIEVE_TOKEN not in output_text:
                continue

            #### Step 3: Perform DB lookup
            return_value = "unknown"
            try:
                split = prompt.rsplit(DB_START_TOKEN)
                query = split[-1]
                return_value = self.db.retrieve_from_database(DB_START_TOKEN  + query, 0.0) #ignoring the threshold, for now using top1 fallback policy
            except Exception as e:
                print(f"Database lookup failed: {e}")


            #### Step 4: Append retrieved value and db_end token
            prompt += return_value + DB_END_TOKEN
        try:
            answer = prompt.split("<answer>")[1].split("<answer/>")[0]
            trace = [AgentStep(prompt, answer, "generate")]
            return answer, trace
        except Exception as e:
            print(f"LMLM was unable to generate a correctly formated answer, receieved error : {e}... Defaulting to empty answer.")
            return "", [AgentStep(prompt, "", "generate")]

         