## 单机w8a8量化

```bash
docker exec -it gl_hy4_001 bash
```

```bash
MODEL="/mnt/weight/Hy4-preview-Testing-w8a8"

NIC="enp209s0f0"
LOCAL_IP="192.168.13.198"
PORT=8000

export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_IF_IP=${LOCAL_IP}
export GLOO_SOCKET_IFNAME=${NIC}
export TP_SOCKET_IFNAME=${NIC}
export HCCL_SOCKET_IFNAME=${NIC}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=128
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 16 \
  --served-model-name hy4_w8a8 \
  --enable-expert-parallel\
  --max-num-seqs 4 \
  --max-model-len 1024 \
  --max-num-batched-tokens 512 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --moe-backend triton \
  --seed 1024 | tee /tmp/vllm_hy4_w8a8.log
```

curl实际结果：

```json
curl -sS http://192.168.13.198:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"model":"hy4_w8a8","messages":[{"role":"user","content":"中国的首都是哪里？"}],"max_tokens":800,"temperature":0}'
{"id":"chatcmpl-b2a6eecb7d384f56","object":"chat.completion","created":1787401729,"model":"hy4_w8a8","choices":[{"index":0,"message":{"role":"assistant","content":"用户问：中国的首都是哪里？\n答案：北京。\n这是一个非常简单的事实性问题，直接回答即可。</think:opensource>中国的首都是**北京**。","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-0.23.0-tp16-ep-88cd1f80","usage":{"prompt_tokens":26,"total_tokens":55,"completion_tokens":29,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null}

curl -sS http://192.168.13.198:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"model":"hy4_w8a8","messages":[{"role":"user","content":"1+1等于几？"}],"max_tokens":300,"temperature":0}'
{"id":"chatcmpl-b4b405df6d521bc9","object":"chat.completion","created":1787401794,"model":"hy4_w8a8","choices":[{"index":0,"message":{"role":"assistant","content":"好的，用户问的是“1+1等于几？”，这是一个非常基础的数学问题。首先，我需要确认用户的需求是什么。可能是一个刚开始学习数学的小孩，或者是在测试我的基本计算能力。不过，不管怎样，这个问题本身很简单，但作为助手，我需要确保回答准确且易于理解。\n\n接下来，我要考虑可能的陷阱。有时候，类似的问题可能有不同的答案，比如在二进制中1+1等于10，或者在布尔代数中可能代表逻辑或。但通常情况下，用户可能只是想知道基本的十进制加法。所以，我需要先确认上下文，但用户没有提供更多信息，所以默认应该是十进制。\n\n然后，我需要组织语言，确保回答清晰。可能需要分步骤解释，比如1加1等于2，或者用实物举例，比如一个苹果加一个苹果等于两个苹果。不过，用户的问题很直接，可能不需要过多解释，但适当的解释可以帮助用户更好地理解。\n\n另外，用户可能是在测试我的响应能力，或者想确认我是否可靠。因此，回答要简洁明了，同时保持友好。比如，可以回答“1+1等于2，这是基本的数学加法运算。”这样既准确又直接。\n\n最后，检查是否有其他可能的解释，比如单位不同，但通常1+1在数学中就是2。确保没有遗漏任何特殊情况，但在这个情况下，应该没有问题。所以，最终答案应该是2，并可能附上简单的解释。</think:opensource>1+1等于2。这是基本的数学加法运算，表示将两个数量合并","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-0.23.0-tp16-ep-88cd1f80","usage":{"prompt_tokens":27,"total_tokens":327,"completion_tokens":300,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null}
```
