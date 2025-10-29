from pathlib import Path
import os
import re
import time
from typing import Optional

from transformers import AutoProcessor

# Don't import vLLM until after NVTE FP8 env overrides are set to avoid early TE picks


class VisionLanguageModel:
    """vLLM-only wrapper for Qwen2.5-VL with text and image support."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", flash_attn: str = "auto"):
        """
        Initialize vLLM backend and a HF processor for chat templates.
        """
        print(f"Initializing vLLM for {model_name}...")

        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        tensor_parallel_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
        # Memory tuning knobs (safe defaults, override via env if needed)
        gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTILIZATION", "0.5"))
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "1536"))
        max_batched_tokens = int(os.environ.get("VLLM_MAX_BATCHED_TOKENS", "2048"))
        max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "2"))
        kv_cache_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE", "auto")
        # Normalize kv cache dtype to vLLM-supported literals
        allowed_kv_dtypes = {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"}
        if kv_cache_dtype in {"fp16", "float16", "bf16"}:
            kv_cache_dtype = "bfloat16"
        elif kv_cache_dtype in {"fp8e4b15", "fp8_e4b15", "e4m3"}:
            kv_cache_dtype = "fp8_e4m3"
        elif kv_cache_dtype in {"fp8e5", "fp8_e5", "e5m2", "fp8", "fp8e4nv", "fp8_e4nv"}:
            kv_cache_dtype = "fp8_e5m2"
        elif kv_cache_dtype not in allowed_kv_dtypes:
            print(f"Unsupported VLLM_KV_CACHE_DTYPE='{kv_cache_dtype}', defaulting to 'bfloat16'")
            kv_cache_dtype = "bfloat16"
        # Allow disabling FlashAttention entirely via env if needed
        enforce_eager_env = os.environ.get("VLLM_ENFORCE_EAGER", "0")
        disable_fa_env = os.environ.get("VLLM_DISABLE_FLASH_ATTN", "0")
        enforce_eager = (enforce_eager_env == "1") or (disable_fa_env == "1")
        # Normalize/validate weight quantization option if provided
        quantization_raw = os.environ.get("VLLM_QUANTIZATION")  # e.g. "fp8_e4m3", "awq", etc.
        quantization = None
        if quantization_raw:
            q = quantization_raw.strip().lower()
            if q in {"fp8", "fp8e4nv", "fp8_e4nv"}:
                q = "fp8_e5m2"
            allowed_quant = {"awq", "gptq", "squeezellm", "rtn", "fp8_e4m3", "fp8_e5m2"}
            if q in allowed_quant:
                quantization = q
            else:
                print(f"Unsupported VLLM_QUANTIZATION='{quantization_raw}', ignoring weight quantization")
        # Ensure Transformer Engine uses supported FP8 dtype names for this arch
        if kv_cache_dtype.startswith("fp8"):
            te_fp8 = "fp8e5" if kv_cache_dtype == "fp8_e5m2" else "fp8e4b15"
            for k in ("NVTE_FP8_DTYPE", "TRANSFORMER_ENGINE_FP8_DTYPE", "TE_FP8_DTYPE"):
                os.environ[k] = te_fp8
        else:
            for k in ("NVTE_FP8_DTYPE", "TRANSFORMER_ENGINE_FP8_DTYPE", "TE_FP8_DTYPE"):
                if k in os.environ:
                    del os.environ[k]

        # Disable video by default to prevent large encoder cache profiling on startup
        limit_mm_image = int(os.environ.get("VLLM_LIMIT_MM_IMAGE", "1"))
        limit_mm_video = int(os.environ.get("VLLM_LIMIT_MM_VIDEO", "0"))
        limit_mm_audio = int(os.environ.get("VLLM_LIMIT_MM_AUDIO", "0"))

        # Import vLLM only after FP8 env overrides are set to avoid early TE picks
        from vllm import LLM, SamplingParams  # type: ignore

        llm_kwargs = dict(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_batched_tokens,
            max_num_seqs=max_num_seqs,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
            limit_mm_per_prompt={
                "image": limit_mm_image,
                "video": limit_mm_video,
                "audio": limit_mm_audio,
            },
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.vllm = LLM(**llm_kwargs)
        self._vllm_sampling_cls = SamplingParams
        print("vLLM backend initialized")

        self.recent_responses = []
        self.system_prompt = ""
        self.MAX_RECENT_RESPONSES = 10
        self.MAX_WORDS_PER_RESPONSE = 50

    def infer(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 1.1,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """
        Run inference with text prompt and optional image using vLLM exclusively.
        """
        start_time = time.time()

        # Build messages in Qwen2.5-VL format with optional system prompt
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": str(self.system_prompt)},
                    ],
                }
            )

        if image_path:
            image_path_obj = Path(image_path)
            if not image_path_obj.is_absolute() and not image_path_obj.exists():
                candidate = Path(__file__).parent / image_path
                if candidate.exists():
                    image_path = str(candidate)
            print(f"Loading image: {image_path}")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            print(f"VLM text input: {prompt}")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            )

        # vLLM path for both text-only and image inputs
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        temperature_val = float(temperature)
        top_p_val = float(top_p)
        if not do_sample or temperature_val == 0.0:
            temperature_val = 0.0

        sampling_params = self._vllm_sampling_cls(
            max_tokens=int(max_new_tokens),
            temperature=temperature_val,
            top_p=top_p_val,
        )

        if image_path:
            request = {
                "prompt": text_prompt,
                "multi_modal_data": {"image": [image_path]},
            }
            gen_start = time.time()
            outputs = self.vllm.generate([request], sampling_params)
        else:
            gen_start = time.time()
            outputs = self.vllm.generate([text_prompt], sampling_params)
        gen_elapsed = time.time() - gen_start

        text_out = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        try:
            self.last_generated_tokens = int(len(outputs[0].outputs[0].token_ids))
        except Exception:
            self.last_generated_tokens = 0
        gen_tps = (self.last_generated_tokens / gen_elapsed) if gen_elapsed > 0 and self.last_generated_tokens > 0 else 0.0
        print(f"vLLM generation: {self.last_generated_tokens} tok in {gen_elapsed:.2f}s = {gen_tps:.2f} tok/s")

        elapsed = time.time() - start_time
        print(f"VLM inference took {elapsed:.2f} seconds")
        return self.postprocess(text_out)


    def postprocess(self, result: str) -> str:
        """
        Ad-hoc postprocessing of the result to remove things that shouldn't be there, but show up sometimes.
        """
        # Remove voice-directions inside asterisks like "*grumble* I'm not happy"
        result = re.sub(r'\*.*?\*', '', result)

        # For some reason Qwen outputs "addCriterion" sometimes, which is just nonsense.  Strip everything after that.
        result = re.sub(r'addCriterion.*', '', result)

        # Remove non-ascii characters.  Don't need it saying "chinese character"
        result = re.sub(r'[^\x00-\x7F]+', '', result)

        # Limit to MAX_WORDS_PER_RESPONSE words.
        words = result.split()
        if len(words) > self.MAX_WORDS_PER_RESPONSE:
            result = " ".join(words[:self.MAX_WORDS_PER_RESPONSE])
            print(f"Truncated response to {len(words)} words to {self.MAX_WORDS_PER_RESPONSE} words")

        # Null out duplicate responses.  (Common with stupid 7B model.)
        if result not in self.recent_responses:
            self.recent_responses.append(result)
            if len(self.recent_responses) > self.MAX_RECENT_RESPONSES:
                self.recent_responses.pop(0)
        else:
            print(f"Ignoring duplicate response: {result}")
            result = ""
        return result

