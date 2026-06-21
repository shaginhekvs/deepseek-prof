from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="facebook/opt-125m",
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.25,
    )
    prompts = [
        "PyTorch profiler shows",
        "NCCL collectives matter because",
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=24, temperature=0.0),
    )
    for out in outputs:
        print(f"PROMPT: {out.prompt}")
        print(f"TEXT: {out.outputs[0].text}")


if __name__ == "__main__":
    main()
