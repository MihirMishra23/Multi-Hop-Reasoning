import json
from agent.agent import Agent, AgentStep
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmlm.database.database_manager import DatabaseManager
from transformers import LogitsProcessor
from lmlm.constants import DB_END_TOKEN, ANSWER_START_TOKEN, DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, ANSWER_END_TOKEN
import os

def _decode_with_special_tokens(outputs, tokenizer, input_len, input_text):
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        if input_text in output_text:
            output_text = output_text.split(input_text)[-1]
        else:
            output_text = tokenizer.decode(outputs[0][input_len:], clean_up_tokenization_spaces=True) 
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
    def __init__(self, model_path = "/share/j_sun/lmlm_multihop/models/Qwen3-1.7B/gemini_sft_v1/_full_ep5_bsz32_new_qa", database_path = "../LMLM/hotpotqa_annotation_results/extracted_database_lookups.json", similarity_threshold = 0.6, adaptive : bool = False, top_k : int = 4):
        self.model_path = model_path
        self.database_path = database_path
        self.top_k = top_k
        metadata_file = os.path.join(os.path.dirname(database_path), "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        self.db = DatabaseManager()
        self.db.load_database(database_path, adaptive= adaptive)
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if self.device=="cuda" else None)
        self.model.to(self.device).eval()
        self.similarity_threshold = similarity_threshold

    def create_prompt_from_query(self, query):
        return f"Question:\n{query}\nAnswer:\n"
    
    def create_prompt_from_query_batch(self, queries : list[str]):
        return [self.create_prompt_from_query(query) for query in queries]

    def run(self, query : str,  index : int, max_tokens = 256, temperature = 0.0):
        count = 0
        prompt = self.create_prompt_from_query(query)

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

            if prompt.endswith(ANSWER_END_TOKEN):
                break

            # Check if <|db_return|> is present
            if DB_RETRIEVE_TOKEN not in output_text:
                continue

            #### Step 3: Perform DB lookup
            return_value = "unknown"
            try:
                split = prompt.rsplit(DB_START_TOKEN)
                db_query = split[-1]
                return_values = self.db.retrieve_from_database(DB_START_TOKEN  + db_query, 0.6, return_triplets = True, top_k = self.top_k) 
                return_value = ", ".join(return_values)
            except Exception as e:
                print(f"Database lookup failed: {e}")


            #### Step 4: Append retrieved value and db_end token
            prompt += return_value + DB_END_TOKEN
        try:
            answer = prompt.split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0]
            if self.metadata:
                golden_triplets =  ", ".join(f"({entity}, {rel}, {val})" for (entity, rel, val) in self.metadata[index]["triplets"])
            else:
                golden_triplets = 'No metadata provided'
            trace = [AgentStep(prompt, answer, "generate", golden_triplets = golden_triplets)]
            return answer, trace
        except Exception as e:
            return "", [AgentStep(prompt, "", "generate")]
        

if __name__ == '__main__':
    #testing script
    agent = LMLMAgent(model_path = "/share/j_sun/lmlm_multihop/models/Qwen3-1.7B/gemini_sft_v1/_full_ep5_bsz32_new_qa", database_path="/share/j_sun/lmlm_multihop/database/gemini/generated_database_validation_42_1000.json")
    for i in range(20):
        answer, trace = agent.run("What is the first two words of the fifth studio album of Joseph Edgar Foreman?")
        # print("answer: \n\n", answer, "\n\n")
        # print("trace: \n\n", trace, "\n\n")
    
