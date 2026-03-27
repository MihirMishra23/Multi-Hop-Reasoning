import json
from agent.agent_class import Agent, AgentStep
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multi_lmlm.database.database_manager import DatabaseManager, build_databases_from_triplets_batch
from transformers import LogitsProcessor
from multi_lmlm.constants import DB_END_TOKEN, ANSWER_START_TOKEN, DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, ANSWER_END_TOKEN
import os
from vllm import LLM, SamplingParams
from trainer.lmlm_basetrainer import LMLMGRPOTrainer
from trl.trainer import GRPOConfig


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
    def __init__(self, model_path = "/share/j_sun/lmlm_multihop/models/Qwen3-1.7B/gemini_sft_v1/_full_ep5_bsz32_new_qa", database_path = None, similarity_threshold = 0.6, adaptive : bool = False, top_k : int = 4, return_triplets : bool = False, use_inverses : bool = False, phase_1: bool = False, batch_size: int = 32, **kwargs ):
        self.model_path = model_path
        self.database_path = database_path
        self.top_k = top_k
        self.phase_1 = phase_1
        self.return_triplets = return_triplets
        self.use_inverses = use_inverses
        self.adaptive = adaptive
        self.batch_size = batch_size

        # Validate inputs
        if not self.phase_1 and self.database_path is None:
            raise ValueError("database_path must be provided when phase_1=False")

        # Initialize trainer for phase_1 mode
        self.trainer = None
        if self.phase_1:
            print(f"Initializing trainer for phase_1 mode with batch_size={batch_size}...")
            tok = AutoTokenizer.from_pretrained(model_path)
            grpo_config = GRPOConfig(
                vllm_mode='colocate',
                vllm_gpu_memory_utilization=0.8,
                temperature=1.3,  # Match GRPO training (was 1.0)
                top_p=0.95,       # Match GRPO training (was 1.0)
                top_k=4,          # Match GRPO training (was 0) - forces diversity
                num_generations=batch_size,
                steps_per_generation=1,
                per_device_train_batch_size=batch_size,
            )
            self.trainer = LMLMGRPOTrainer(
                model=model_path,
                processing_class=tok,
                reward_funcs=[],
                two_phase=True,
                lmlm_database_path=None,
                args=grpo_config,
            )
            print("Trainer initialized for phase_1 mode")

        # Load database (skip if phase_1 since we'll build dynamically)
        if not self.phase_1:
            metadata_file = os.path.join(os.path.dirname(database_path), "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = None
        else:
            self.metadata = None

        if use_inverses:
            print("USING INVERSES WOOHOO")

        self.db = DatabaseManager()
        if not self.phase_1:
            self.db.load_database(database_path, adaptive=adaptive, use_inverses=use_inverses, top_k=self.top_k)

        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path)

        self.similarity_threshold = similarity_threshold

        # Initialize vLLM for batch generation
        self.stop_token_ids = [self.tok.eos_token_id, self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]]
        self.db_retrieve_token_id = self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]
        self.answer_end_token_id = self.tok.encode(ANSWER_END_TOKEN, add_special_tokens=False)[0]


        # Add validation in __init__
        print(f"EOS token ID: {self.tok.eos_token_id}")
        print(f"DB_RETRIEVE_TOKEN ID: {self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]}")
        print(f"Vocab size: {len(self.tok)}")

        # Ensure they're within vocab bounds
        def check_token_in_vocab(tokenizer):
            special_tokens = [DB_START_TOKEN, DB_END_TOKEN, DB_RETRIEVE_TOKEN, 
                  ANSWER_START_TOKEN, ANSWER_END_TOKEN, DB_SEP_TOKEN]
            for token in special_tokens:
                encoded = tokenizer.encode(token, add_special_tokens=False)
                print(f"{token}: {encoded}")
                assert len(encoded) > 0, f"Token {token} not in vocabulary"

        check_token_in_vocab(self.tok)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            # max_model_len=16384,
            seed=42,
            tokenizer=model_path,
        )
        check_token_in_vocab(self.llm.get_tokenizer())

        self.max_turns = 16

    def create_prompt_from_query(self, query):
        return f"Question:\n{query}\nAnswer:\n"
    
    def create_prompt_from_query_batch(self, queries : list[str]):
        return [self.create_prompt_from_query(query) for query in queries]


    def run(self, queries: list[str], indices: list[int] | None = None, max_tokens=256, temperature=0.0, golden_contexts: list[list[str]] | None = None):
        """
        Run batch inference using vLLM with database lookups.

        Args:
            queries: List of questions to answer
            indices: List of indices corresponding to each query (for metadata lookup)
            max_tokens: Maximum tokens to generate per turn before hitting a stop token
            temperature: Sampling temperature (0.0 = greedy)
            golden_contexts: List of golden contexts per query (for phase_1 mode)

        Returns:
            List of (answer, trace) tuples, one per query
        """
        # Build per-question databases from golden contexts if phase_1 mode is enabled
        per_question_dbs = None
        print("golden contexts is None: ", golden_contexts is None)
        if self.phase_1 and golden_contexts is not None:
            # print(f"Building {len(golden_contexts)} per-question databases from golden contexts...")
            # print("\n\n\n\ngolden contexts is : ", golden_contexts)
            # print("queries is : ", queries , "\n\n\n")

            # Generate triplets for the entire batch (returns list of lists, one per question)
            triplets_per_question = self.trainer._generate_two_phase(
                qa_prompts=queries,
                contexts=golden_contexts,
                fast_build_db=True,
                return_per_question_dbs=True
            )
            print("Triplets per questions is :", triplets_per_question)
            total_triplets = sum(len(t) for t in triplets_per_question)
            print(f"Generated {total_triplets} total triplets across {len(triplets_per_question)} questions")
            if len(triplets_per_question) > 0:
                print(f"  First question: {len(triplets_per_question[0])} triplets")

            # Build databases using batch utility (efficient embedding computation)
            per_question_dbs = build_databases_from_triplets_batch(
                triplets_per_question,
                top_k=self.top_k,
                default_threshold=self.similarity_threshold,
                adaptive=self.adaptive,
                use_inverses=self.use_inverses,
            )
            print(f"Built {len(per_question_dbs)} per-question databases")

        # Create initial prompts for all queries
        prompts = self.create_prompt_from_query_batch(queries)

        # Track which queries are still generating
        active = [True] * len(queries)
        results = [(None, None) for _ in range(len(queries))]
        self._lookup_logs = [[] for _ in range(len(queries))]

        # Generation loop - continue until all queries complete
        # Max iterations to prevent infinite loops (in case of malformed outputs)
        
        # BUG: potential bug here. need to check the prompt length each turn to make sure it doesn't exceed the max model length
        for turn in range(self.max_turns):
            # Only generate for active queries
            active_prompts = [p for i, p in enumerate(prompts) if active[i]]
            if not active_prompts:
                break
    
            # DEBUG: Check prompt lengths
            for idx, prompt in enumerate(active_prompts):
                prompt_len = len(self.tok.encode(prompt))
                max_len = self.llm.llm_engine.model_config.max_model_len
                if prompt_len + max_tokens > max_len:
                    print(f"Warning: Prompt {idx} length {prompt_len} + max_tokens {max_tokens} exceeds max {max_len}")
                    # Either truncate or mark as inactive
                    active[idx] = False
                    continue

            # Setup sampling parameters with stop tokens
            # vLLM will automatically stop at DB_RETRIEVE_TOKEN or EOS
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                max_tokens=max_tokens,  # Generate up to max_tokens or until stop token
                stop_token_ids=self.stop_token_ids,
                # logprobs=0, # help solve the bug of illegal cuda memory access
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
                    lookup_log = {
                        "query": None,
                        "success": False,
                        "returned_count": 0,
                        "error": None,
                    }
                    try:
                        split = prompts[i].rsplit(DB_START_TOKEN)
                        db_query = split[-1]
                        lookup_log["query"] = db_query

                        # Use per-question database if in phase_1 mode, otherwise use shared db
                        db_to_use = per_question_dbs[i] if per_question_dbs else self.db

                        # print("\n\nquery is ", DB_START_TOKEN + db_query)
                        # print("db triplets is :", db_to_use.database["triplets"])

                        return_values = db_to_use.retrieve_from_database(
                            DB_START_TOKEN + db_query,
                            self.similarity_threshold,
                            return_triplets=self.return_triplets,
                            top_k=self.top_k
                        )
                        # print("returneed values is :", return_values)
                        lookup_log["returned_count"] = len(return_values)
                        lookup_log["success"] = len(return_values) > 0
                        return_value = ", ".join(return_values)
                    except Exception as e:
                        lookup_log["error"] = str(e)
                    self._lookup_logs[i].append(lookup_log)

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
    import argparse
    from datetime import datetime
    from eval.metrics import exact_match_score, f1_score

    parser = argparse.ArgumentParser(description='Run two-phase evaluation from JSON file')
    parser.add_argument('--input-file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output-file', type=str, default=None, help='Path to output JSON file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--top-k', type=int, default=4, help='Top-k for database retrieval')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold for retrieval')
    parser.add_argument('--use-inverses', action='store_true', help='Use inverse relationships in database')
    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Extract phase1 and phase2 prompts
    phase1_prompts = [item['phase1_prompt'] for item in data]
    phase2_prompts = [item['phase2_prompt'] for item in data]
    gold_answers = [item['answer'] for item in data]

    # Phase 1: Extract triplets
    print(f"\n{'='*80}")
    print(f"Phase 1: Extracting triplets")
    print(f"{'='*80}\n")

    agent_phase1 = LMLMAgent(
        model_path=args.model_path,
        database_path=None,
        phase_1=True,
        batch_size=args.batch_size,
        top_k=args.top_k,
        return_triplets=False,
        use_inverses=args.use_inverses
    )

    triplet_outputs, _ = agent_phase1.run(phase1_prompts)
    print(f"Extracted triplets for {len(triplet_outputs)} examples")

    # Build databases from triplets
    print(f"\n{'='*80}")
    print(f"Building databases from triplets")
    print(f"{'='*80}\n")

    databases = []
    for i, triplet_output in enumerate(triplet_outputs):
        db = DatabaseManager(use_inverses=args.use_inverses)
        db.add_triplets_from_text(triplet_output)
        databases.append(db)
        if (i + 1) % 100 == 0:
            print(f"Built {i + 1}/{len(triplet_outputs)} databases")

    # Phase 2: Answer questions
    print(f"\n{'='*80}")
    print(f"Phase 2: Answering questions")
    print(f"{'='*80}\n")

    agent_phase2 = LMLMAgent(
        model_path=args.model_path,
        database_path=None,
        phase_1=False,
        batch_size=args.batch_size,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        return_triplets=False,
        use_inverses=args.use_inverses
    )

    predictions, traces = agent_phase2.run(phase2_prompts, per_question_dbs=databases)

    # Calculate scores
    print(f"\n{'='*80}")
    print(f"Calculating scores")
    print(f"{'='*80}\n")

    exact_matches = []
    f1_scores = []
    results = []

    for i, (pred, gold, trace) in enumerate(zip(predictions, gold_answers, traces)):
        em = exact_match_score(pred, gold)
        f1, precision, recall = f1_score(pred, gold)

        exact_matches.append(em)
        f1_scores.append(f1)

        results.append({
            'index': i,
            'phase1_prompt': phase1_prompts[i],
            'phase2_prompt': phase2_prompts[i],
            'triplets_extracted': triplet_outputs[i],
            'prediction': pred,
            'gold_answer': gold,
            'exact_match': em,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'trace': [{'prompt': step.prompt, 'answer': step.answer, 'action': step.action} for step in trace]
        })

    avg_em = sum(exact_matches) / len(exact_matches) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100

    print(f"Exact Match: {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")

    # Save results
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_file = f"two_phase_results_{timestamp}.json"

    output_data = {
        'metadata': {
            'model_path': args.model_path,
            'input_file': args.input_file,
            'batch_size': args.batch_size,
            'top_k': args.top_k,
            'threshold': args.threshold,
            'use_inverses': args.use_inverses,
            'total_examples': len(data),
            'exact_match': avg_em,
            'f1_score': avg_f1
        },
        'results': results
    }

    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Done!")
    



