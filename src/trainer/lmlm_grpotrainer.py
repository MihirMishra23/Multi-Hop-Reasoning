from trl.trainer.grpo_trainer import GRPOTrainer
from multi_lmlm.database.database_manager import DatabaseManager
from multi_lmlm.constants import DB_START_TOKEN, DB_END_TOKEN, DB_RETRIEVE_TOKEN
import copy
import torch

def extract_db_lookup(text: str) -> str | None:
    """Extract database lookup from text."""
    if DB_START_TOKEN in text and DB_RETRIEVE_TOKEN in text:
        return DB_START_TOKEN + text.split(DB_START_TOKEN)[1].split(DB_RETRIEVE_TOKEN)[0] + DB_RETRIEVE_TOKEN
    else:
        return None


class LMLMGRPOTrainer(GRPOTrainer):
    """
    Extended GRPO Trainer with LMLM database integration for tool-augmented training.
    """

    def __init__(
        self,
        model,
        reward_funcs,
        lmlm_database_path: str,
        adaptive_k: bool = False,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        tools=None,
        rollout_func=None,
    ):
        # Initialize LMLM database before calling super().__init__
        self.db = DatabaseManager()
        self.db.load_database(lmlm_database_path, adaptive=adaptive_k)
        
        # Call parent constructor
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            tools=None,  # <-- Pass None to skip parent's tool initialization
            rollout_func=rollout_func,
        )
        self.tools = tools

        # Override stop_token_ids to include DB_RETRIEVE_TOKEN
        self.stop_token_ids = [
            self.processing_class.eos_token_id,
            self.processing_class.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]
        ]
        
        # Store DB-specific token IDs
        self.db_retrieve_token_id = self.processing_class.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]
        self.db_end_token_id = self.processing_class.encode(DB_END_TOKEN, add_special_tokens=False)[0]

    def _generate_single_turn(self, prompts: list):
        """Override to add stop_token_ids parameter for vLLM generation."""
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up(tags=["weights"])
                self.llm.collective_rpc("reload_weights")

            # Update vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Handle conversational format
            if is_conversational({"prompt": prompts[0]}):
                from trl.data_utils import prepare_multimodal_messages_vllm
                prompts = [prepare_multimodal_messages_vllm(prompt) for prompt in prompts]

            # Convert tool call arguments to JSON strings for vLLM
            for prompt in prompts:
                if is_conversational({"prompt": prompt}):
                    for message in prompt:
                        if "tool_calls" in message:
                            for call in message["tool_calls"]:
                                args = call["function"]["arguments"]
                                if isinstance(args, dict):
                                    import json
                                    call["function"]["arguments"] = json.dumps(args)

            # Generate with vLLM
            if self.vllm_mode == "server":
                # Server mode implementation (same as parent)
                return super()._generate_single_turn(prompts)
                
            elif self.vllm_mode == "colocate":
                if self.rollout_func is not None:
                    # Use custom rollout function if provided
                    return super()._generate_single_turn(prompts)
                else:
                    from vllm import SamplingParams
                    
                    guided_decoding = None
                    
                    generation_kwargs = {
                        "n": 1,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "guided_decoding": guided_decoding,
                        "logprobs": 0,
                        "stop_token_ids": self.stop_token_ids,  # <-- LMLM MODIFICATION: Add custom stop tokens
                    }
                    if self.args.generation_kwargs is not None:
                        generation_kwargs.update(self.args.generation_kwargs)
                    sampling_params = SamplingParams(**generation_kwargs)

                    # Handle tensor parallel gathering
                    if self.vllm_tensor_parallel_size > 1:
                        orig_size = len(prompts)
                        gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                        all_prompts = [p for sublist in gathered_prompts for p in sublist]
                    else:
                        all_prompts = prompts

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.wake_up(tags=["kv_cache"])

                    # Generate completions
                    from trl.data_utils import is_conversational
                    if is_conversational({"prompt": prompts[0]}):
                        all_outputs = self.llm.chat(
                            all_prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                            chat_template_kwargs=self.chat_template_kwargs,
                            tools=self.tools,
                            chat_template=self.chat_template,
                        )
                    else:
                        all_outputs = self.llm.generate(
                            all_prompts, sampling_params=sampling_params, use_tqdm=False
                        )

                    all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                    all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                    all_logprobs = [
                        [next(iter(lp.values())).logprob for lp in output.logprobs]
                        for outputs in all_outputs
                        for output in outputs.outputs
                    ]

                    # Slice for this rank
                    if self.vllm_tensor_parallel_size > 1:
                        local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                        tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                        prompt_ids = all_prompt_ids[tp_slice]
                        completion_ids = all_completion_ids[tp_slice]
                        logprobs = all_logprobs[tp_slice]
                    else:
                        prompt_ids = all_prompt_ids
                        completion_ids = all_completion_ids
                        logprobs = all_logprobs

                    extra_fields = {}

                    if self.args.vllm_enable_sleep_mode:
                        self.llm.sleep(level=2)
                        
                    return prompt_ids, completion_ids, logprobs, extra_fields
        else:
            # Non-vLLM generation (transformers)
            return super()._generate_single_turn(prompts)

    # def _generate_and_score_completions(
    #     self, inputs: list[dict[str, torch.Tensor | Any]]
    # ) -> dict[str, torch.Tensor | Any]:
    #     """Override to decode completions without skipping special tokens."""
    #     # Call parent method to do most of the work
    #     result = super()._generate_and_score_completions(inputs)
        
    #     # The only change: re-decode completions with skip_special_tokens=False
    #     # This preserves DB tokens in the decoded text
    #     completion_ids = result["completion_ids"]
    #     completions_text = self.processing_class.batch_decode(
    #         completion_ids, skip_special_tokens=False  # <-- MODIFICATION: Keep special tokens
    #     )
        
    #     # Note: The parent already stored these in self._logs, 
    #     # but if you need the updated versions, you can update them here
    #     # However, _generate_and_score_completions doesn't return completions_text,
    #     # so this might not affect the training loop directly.
    #     # The real completions used in rewards are from the _generate method.
        
    #     return result

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs):
        """Implement LMLM database lookup logic."""
        tool_calls = [extract_db_lookup(completion) for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_call_count = 0
        tool_failure_count = 0

        while idxs_with_tool:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]

            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call = tool_calls[idx]

                prompt_completion_tools[idx] += completions[idx_with_tool]
                tool_call_count += 1
                
                try:
                    result = ", ".join(self.db.retrieve_from_database(tool_call)) + DB_END_TOKEN
                except Exception as e:
                    print(f"DB lookup failed: {str(e)}")
                    tool_failure_count += 1
                    result = "unknown" + DB_END_TOKEN

                prompt_completion_tools[idx] += result
                completions[idx_with_tool] += result
                value_ids = self.processing_class(result, add_special_tokens=False)["input_ids"]
                completion_ids[idx_with_tool] += value_ids
                logprobs[idx_with_tool] += [0.0] * len(value_ids)

            # Generate new completions after tool execution
            prompt_completion_tool_ids, post_tool_ids, post_tool_logprobs, _ = self._generate_single_turn(
                prompt_completion_tools
            )

            # Verify lengths
            for i in range(len(post_tool_logprobs)):
                assert len(post_tool_ids[i]) == len(post_tool_logprobs[i]), \
                    f"Length mismatch: post_tool_ids={len(post_tool_ids[i])}, post_tool_logprobs={len(post_tool_logprobs[i])}"

            # Update completion_ids and logprobs
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_ids[idx_with_tool] += post_tool_ids[idx]
                if logprobs is not None:
                    logprobs[idx_with_tool] += post_tool_logprobs[idx]
                    assert len(logprobs[idx_with_tool]) == len(completion_ids[idx_with_tool]), \
                        f"Length mismatch after update: logprobs={len(logprobs[idx_with_tool])}, completion_ids={len(completion_ids[idx_with_tool])}"

            # Decode post-tool completions
            post_tool_completions = [
                self.processing_class.decode(ids, skip_special_tokens=False) if ids else "" 
                for ids in post_tool_ids
            ]

            # Add post-tool completions to existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:
                    completions[idx_with_tool] += post_tool_completions[idx]

            # Check for further tool calls
            tool_calls = [extract_db_lookup(completion) for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]

        # Create tool mask (0 for DB content, 1 for model-generated content)
        tool_mask = [[] for _ in range(len(completion_ids))]
        for j in range(len(completion_ids)):
            mask_val = 1
            for i in completion_ids[j]:
                if i == self.db_retrieve_token_id:
                    mask_val = 0
                    tool_mask[j].append(1)  # The retrieve token itself is included
                    continue
                if i == self.db_end_token_id:
                    mask_val = 1
                    tool_mask[j].append(0)  # The end token itself is excluded
                    continue
                tool_mask[j].append(mask_val)

        # Final verification
        for i in range(len(completion_ids)):
            assert len(logprobs[i]) == len(completion_ids[i]), \
                f"Final length mismatch: logprobs={len(logprobs[i])}, completion_ids={len(completion_ids[i])}"
            
        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count