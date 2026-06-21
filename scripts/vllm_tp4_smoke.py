from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="facebook/opt-125m",
        dtype="float16",
        max_model_len=512,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.20,
    )
    outputs = llm.generate(
        [
            "Tensor parallel inference uses",
            "Expert parallel routing in large MoE models",
        ],
        SamplingParams(max_tokens=24, temperature=0.0),
    )
    for out in outputs:
        print(f"PROMPT: {out.prompt}")
        print(f"TEXT: {out.outputs[0].text}")


if __name__ == "__main__":
    main()
