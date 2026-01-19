import json
from agent.agent import Agent, AgentStep
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multi_lmlm.database.database_manager import DatabaseManager
from transformers import LogitsProcessor
from multi_lmlm.constants import DB_END_TOKEN, ANSWER_START_TOKEN, DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, ANSWER_END_TOKEN
import os
from vllm import LLM, SamplingParams
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

        # Initialize vLLM for batch generation
        self.stop_token_ids = [self.tok.eos_token_id, self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]]
        self.db_retrieve_token_id = self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]
        self.answer_end_token_id = self.tok.encode(ANSWER_END_TOKEN, add_special_tokens=False)[0]

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            seed=42,
        )

    def create_prompt_from_query(self, query):
        return f"Question:\n{query}\nAnswer:\n"
    
    def create_prompt_from_query_batch(self, queries : list[str]):
        return [self.create_prompt_from_query(query) for query in queries]


    def run(self, queries: list[str], indices: list[int] | None = None, max_tokens=256, temperature=0.0):
        """
        Run batch inference using vLLM with database lookups.

        Args:
            queries: List of questions to answer
            indices: List of indices corresponding to each query (for metadata lookup)
            max_tokens: Maximum tokens to generate per turn before hitting a stop token
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            List of (answer, trace) tuples, one per query
        """
        # Create initial prompts for all queries
        prompts = self.create_prompt_from_query_batch(queries)

        # Track which queries are still generating
        active = [True] * len(queries)
        results = [(None, None) for _ in range(len(queries))]

        # Generation loop - continue until all queries complete
        # Max iterations to prevent infinite loops (in case of malformed outputs)
        max_turns = 16
        for turn in range(max_turns):
            # Only generate for active queries
            active_prompts = [p for i, p in enumerate(prompts) if active[i]]
            if not active_prompts:
                break

            # Setup sampling parameters with stop tokens
            # vLLM will automatically stop at DB_RETRIEVE_TOKEN or EOS
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                max_tokens=max_tokens,  # Generate up to max_tokens or until stop token
                stop_token_ids=self.stop_token_ids,
                logprobs=0,
            )

            # Generate for all active prompts
            outputs = self.llm.generate(active_prompts, sampling_params=sampling_params, use_tqdm=False)

            # Process outputs and update prompts
            active_idx = 0
            for i in range(len(queries)):
                if not active[i]:
                    continue

                output = outputs[active_idx]
                active_idx += 1

                # Extract generated text
                if len(output.outputs) > 0 and len(output.outputs[0].token_ids) > 0:
                    generated_tokens = output.outputs[0].token_ids
                    generated_text = self.tok.decode(generated_tokens, skip_special_tokens=False)
                    prompts[i] += generated_text
                else:
                    # No tokens generated, likely hit stop token immediately
                    generated_text = ""

                # Check if this query has completed
                if ANSWER_END_TOKEN in prompts[i]:
                    active[i] = False
                    try:
                        answer = prompts[i].split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0]
                        if self.metadata and indices and indices[i] < len(self.metadata):
                            golden_triplets = ", ".join(
                                f"({entity}, {rel}, {val})"
                                for (entity, rel, val) in self.metadata[indices[i]]["triplets"]
                            )
                        else:
                            golden_triplets = 'No metadata provided'
                        trace = [AgentStep(prompts[i], answer, "generate", golden_triplets=golden_triplets)]
                        results[i] = (answer, trace)
                    except Exception as e:
                        results[i] = ("", [AgentStep(prompts[i], "", "generate")])
                    continue

                # Check if we need to perform database lookup
                if DB_RETRIEVE_TOKEN in generated_text:
                    # Extract database query
                    return_value = "unknown"
                    try:
                        split = prompts[i].rsplit(DB_START_TOKEN)
                        db_query = split[-1]
                        return_values = self.db.retrieve_from_database(
                            DB_START_TOKEN + db_query,
                            self.similarity_threshold,
                            return_triplets=False,
                            top_k=self.top_k
                        )
                        return_value = ", ".join(return_values)
                    except Exception as e:
                        print(f"Database lookup failed: {e}")

                    # Append retrieved value and db_end token
                    prompts[i] += return_value + DB_END_TOKEN

        # Finalize any queries that didn't complete
        for i in range(len(queries)):
            if results[i][0] is None:
                results[i] = ("", [AgentStep(prompts[i], "", "generate")])

        answers = [r[0] for r in results]
        traces = [r[1] for r in results]
        return answers, traces

if __name__ == '__main__':
    #testing script
    agent = LMLMAgent(model_path = "/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48_th0.8-grpo-g8-bs16-s8-b0.0-ep5-n8000/checkpoint-1000", database_path="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json")
    for i in range(5):
        results = agent.run_vllm(["The Twelfth United States Army Group commander was the first chairman of what?"], 0)
        print("results :\n\n" , results)
    



