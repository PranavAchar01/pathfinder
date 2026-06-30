# Dedicated vLLM pod serving Qwen with an OpenAI-compatible API on :8000.
FROM vllm/vllm-openai:latest

ENV MODEL=Qwen/Qwen2.5-0.5B-Instruct

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "Qwen/Qwen2.5-0.5B-Instruct", \
            "--host", "0.0.0.0", "--port", "8000", \
            "--max-model-len", "8192", "--trust-remote-code"]
