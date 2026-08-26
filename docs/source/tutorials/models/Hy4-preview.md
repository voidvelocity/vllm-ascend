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


## PD分离拉起

### P

```bash
#!/bin/bash
# Node0 (192.168.13.198) - HY4 PD 分离部署 Prefill 节点 (1P1D, kv_producer)
# 参考: vllm-ascend/docs/source/tutorials/models/DeepSeek-V3.1.md "Prefill-Decode Disaggregation"
# 形态: 1P1D; 本机 = P 节点, TP=16 整机 16 卡, 仅做 prefill, KV 经 MooncakeConnectorV1 (RDMA) 传给 D 节点
# 说明: HY4 不用 MooncakeLayerwiseConnector —— HY4 78 层 DSA 每层 k/v/dsa/scale 独立 2MB 对齐分配,
#       每卡共 354 个 KV 内存区域, 超过 HCCL 单进程 256 个注册区域上限; 且每层双 attn 模块
#       (self_attn.attn + self_attn.npu_attn.attn) 触发 layerwise 连接器 "multiple attn_module" 断言.
# 权重: /mnt/weight/Hy4-preview-Testing-w8a8-v2 (NFS 共享, 两节点路径一致)
# 容器执行: 宿主网络容器内运行; 需已安装 Mooncake (python -c "from mooncake.engine import TransferEngine" 可通过)
#           若依赖缺失参考 docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md 编译安装
#
# 启动顺序 (1P1D):
#   1) 本节点 (198) 执行本脚本 (P, engine port 9100)
#   2) D 节点 (197) 容器内执行 run_d_node.sh
#   3) 两节点均出现 "Application startup complete" 后, 在本节点执行 run_proxy.sh
#   4) 客户端请求统一发往 proxy: http://192.168.13.198:8000/v1/chat/completions
#
# 注意: w8a8-v2 量化权重不含 MTP 结构权重 (enorm/eh_proj/hnorm/decoder_layer.* 等), 不启用 --speculative-config

set -u

find /vllm-workspace/vllm -type d -name "__pycache__" -exec rm -r {} +
find /vllm-workspace/vllm-ascend -type d -name "__pycache__" -exec rm -r {} +
rm -rf ~/.triton/cache

# === 节点网络配置 (按实际环境调整网卡名与 IP) ===
NIC="enp209s0f0"
LOCAL_IP="192.168.13.198"      # 本机 IP (P 节点)
DECODE_IP="192.168.13.197"     # D 节点 IP
ENGINE_PORT=9201               # 本节点 vllm engine 端口 (供 proxy 转发, 不直接对客户端)
KV_PORT=36000                  # Mooncake kv_port; 16 卡/节点须 >= 36000, 规避 RDMA 随机端口 [20000, 36000)

# === 服务配置 ===
MODEL="/mnt/weight/Hy4-preview-Testing-w8a8-v2"
SERVED_NAME="hy4_w8a8_v2"
MAX_MODEL_LEN=512             # 必须 >= prompt+max_tokens: proxy 只对 P 改写 max_tokens=1, D 按原始 max_tokens 校验, 过小 D 会 400
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=32    # prefill 吞吐关键参数, 可按压测逐步上调 (DeepSeek PD 参考值 32560)

# === 环境变量 (通信接口 + vllm-ascend 关键开关, 与 HY4 单机/双机脚本一致) ===
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
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# === PD 分离 (MooncakeConnectorV1) 相关, 参考 DeepSeek-V3.1 PD 部署 ===
# 注意: ASCEND_AGGREGATE_ENABLE / ASCEND_TRANSPORT_PRINT / ACL_OP_INIT_MODE / ASCEND_A3_ENABLE
#       为 V3.2 layerwise 专属开关, V1 (请求级传输) 不需要
# P 节点自动释放请求 KV cache 的超时 (秒)
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1   # HY4 实测未启用, 与单机/双机脚本保持一致
# Mooncake 库路径 (若 import 失败时按实际安装路径补充, V3.1 文档参考值如下):
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

# === 软链工作区 (幂等) ===
ln -sfn /mnt/share/gengli/vllm_hy/vllm /vllm-workspace/vllm
ln -sfn /mnt/share/gengli/vllm_hy/vllm-ascend /vllm-workspace/vllm-ascend

# === 杀旧 vllm ===
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "EngineCore" 2>/dev/null
sleep 3

# === 启动 vllm (P 节点, kv_producer) ===
# 与 DeepSeek-V3.1 PD 参考对齐:
#   - --enforce-eager: P 节点无需图捕获, 拉起更快
#   - 大 max-num-batched-tokens: prefill 每 step 吃满 token 预算
#   - kv_connector_extra_config 描述 1P1D 全局拓扑: prefill/decode 各 dp=1, tp=16
# DP=1 单实例, 无需 launch_online_dp.py 与 --data-parallel-* 参数
exec vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${ENGINE_PORT} \
  --tensor-parallel-size 16 \
  --served-model-name ${SERVED_NAME} \
  --enable-expert-parallel \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --gpu-memory-utilization 0.90 \
  --moe-backend triton \
  --seed 1024 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_producer", "kv_port": "'"${KV_PORT}"'", "kv_connector_extra_config": {"prefill": {"dp_size": 1, "tp_size": 16}, "decode": {"dp_size": 1, "tp_size": 16}}}' \
  | tee /mnt/share/gengli/vllm_hy/GoodProj/test_results/debug/p_diag.log

```

### D

```bash
#!/bin/bash
# Node1 (192.168.13.197) - HY4 PD 分离部署 Decode 节点 (1P1D, kv_consumer)
# 参考: vllm-ascend/docs/source/tutorials/models/DeepSeek-V3.1.md "Prefill-Decode Disaggregation"
# 形态: 1P1D; 本机 = D 节点, TP=16 整机 16 卡, 仅做 decode, 通过 MooncakeConnectorV1 接收 P 节点 KV
# 说明: HY4 不用 MooncakeLayerwiseConnector —— KV 注册区域数 354 超 HCCL 256 上限, 且每层双 attn 模块
#       触发 layerwise 连接器断言 (详见 run_p_node.sh 头部说明)
# 权重: /mnt/weight/Hy4-preview-Testing-w8a8-v2 (NFS 共享, 与 P 节点路径一致)
# 容器执行: 宿主网络容器内运行; 需已安装 Mooncake (python -c "from mooncake.engine import TransferEngine" 可通过)
#
# 前置条件: P 节点 (198) 已先执行 run_p_node.sh, 且已出现 "Application startup complete"
# 客户端不直接访问本节点, 请求经 proxy (198:8000) 转发
#
# 注意: w8a8-v2 量化权重不含 MTP 结构权重, 不启用 --speculative-config
#       因此 decode 每 step 每序列 1 token: capture_sizes=[1,2,4,8], batched_tokens=16 (DeepSeek 带MTP2 为 4x3=12)

set -u

ulimit -l unlimited   # 避免 cudagraph 捕获时 host pinned memory OOM

find /vllm-workspace/vllm -type d -name "__pycache__" -exec rm -r {} +
find /vllm-workspace/vllm-ascend -type d -name "__pycache__" -exec rm -r {} +
rm -rf ~/.triton/cache

# === 节点网络配置 (按实际环境调整网卡名与 IP) ===
NIC="enp209s0f0"
LOCAL_IP="192.168.13.197"      # 本机 IP (D 节点)
PREFILL_IP="192.168.13.198"    # P 节点 IP
ENGINE_PORT=9201               # 本节点 vllm engine 端口 (供 proxy 转发)
KV_PORT=36100                  # Mooncake kv_port; 与 P 节点错开便于排查; 16 卡/节点须 >= 36000

# === 服务配置 ===
MODEL="/mnt/weight/Hy4-preview-Testing-w8a8-v2"
SERVED_NAME="hy4_w8a8_v2"
MAX_MODEL_LEN=512              # 与 P 节点一致; 且必须 >= prompt+max_tokens (D 按原始 max_tokens 校验, 超限 D 返回 400)
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=4       # decode-only: 8 seqs x 1 token/step, 2x 余量

# === 环境变量 (通信接口 + vllm-ascend 关键开关, 与 P 节点一致) ===
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_IF_IP=${LOCAL_IP}
export GLOO_SOCKET_IFNAME=${NIC}
export TP_SOCKET_IFNAME=${NIC}
export HCCL_SOCKET_IFNAME=${NIC}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=128       # DeepSeek PD 参考为 256/600, HY4 实测 128, 可按压测调优
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# === PD 分离 (MooncakeConnectorV1) 相关 ===
# 注意: ASCEND_AGGREGATE_ENABLE / ASCEND_TRANSPORT_PRINT / ACL_OP_INIT_MODE / ASCEND_A3_ENABLE
#       为 V3.2 layerwise 专属开关, V1 (请求级传输) 不需要
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480
export TASK_QUEUE_ENABLE=1     # D 节点开启任务队列 (DeepSeek-V3.1 PD decode 侧要求)
# Mooncake 库路径 (若 import 失败时按实际安装路径补充, V3.1 文档参考值如下):
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

# === 软链工作区 (幂等) ===
ln -sfn /mnt/share/gengli/vllm_hy/vllm /vllm-workspace/vllm
ln -sfn /mnt/share/gengli/vllm_hy/vllm-ascend /vllm-workspace/vllm-ascend

# === 杀旧 vllm ===
pkill -9 -f "vllm serve" 2>/dev/null
pkill -9 -f "EngineCore" 2>/dev/null
sleep 3

# === 启动 vllm (D 节点, kv_consumer) ===
# 与 DeepSeek-V3.1 PD 参考对齐:
#   - FULL_DECODE_ONLY 图捕获: 仅 decode 走 cudagraph, 加速小 batch 续写
#   - recompute_scheduler_enable: KV 传输异常时 D 侧兜底重算
#   - kv_connector_extra_config 与 P 节点完全一致 (描述 1P1D 全局拓扑)
exec vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${ENGINE_PORT} \
  --tensor-parallel-size 16 \
  --served-model-name ${SERVED_NAME} \
  --enable-expert-parallel \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --gpu-memory-utilization 0.85 \
  --moe-backend triton \
  --seed 1024 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]}' \
  --additional-config '{"recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeConnectorV1", "kv_role": "kv_consumer", "kv_port": "'"${KV_PORT}"'", "kv_connector_extra_config": {"prefill": {"dp_size": 1, "tp_size": 16}, "decode": {"dp_size": 1, "tp_size": 16}}}' \
  | tee /mnt/share/gengli/vllm_hy/GoodProj/test_results/debug/d_diag.log
```

proxy
```bash
#!/bin/bash
# HY4 PD 分离部署 请求转发代理 (在 P 节点 192.168.13.198 容器内执行)
# 参考: vllm-ascend/docs/source/tutorials/models/DeepSeek-V3.1.md "Request Forwarding"
# 作用: 客户端统一请求 proxy (8000), proxy 分发请求到 P/D 实例;
#       P 节点 prefill 完成后 KV 经 Mooncake 传至 D 节点续写, D 侧流式返回
# 前置: P 节点 (run_p_node.sh) 与 D 节点 (run_d_node.sh) 均已 "Application startup complete"
# 注意: MooncakeConnectorV1 (请求级传输) 配套 load_balance_proxy_server_example.py;
#       layerwise 版专用 load_balance_proxy_layerwise_server_example.py (HY4 不适用, 见 run_p_node.sh 头部说明)

set -u

# === 后端实例配置 (与 run_p_node.sh / run_d_node.sh 对齐) ===
PREFILL_IP="192.168.13.198"
DECODE_IP="192.168.13.197"
ENGINE_PORT=9201
PROXY_PORT=8000                # 客户端访问入口

PROXY_SCRIPT="/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"

# 代理进程本身不走 http 代理, 避免转发请求被劫持
unset http_proxy
unset https_proxy

# === 软链工作区 (幂等, 确保代理脚本路径可达) ===
ln -sfn /mnt/share/gengli/vllm_hy/vllm /vllm-workspace/vllm
ln -sfn /mnt/share/gengli/vllm_hy/vllm-ascend /vllm-workspace/vllm-ascend

# === 启动 proxy ===
# 1P1D: 1 个 prefiller + 1 个 decoder
# 多 P/D 扩展时, 在 --prefiller-hosts/--decoder-hosts 后追加对应 IP/端口即可
exec python3 ${PROXY_SCRIPT} \
  --host 0.0.0.0 \
  --port ${PROXY_PORT} \
  --prefiller-hosts ${PREFILL_IP} \
  --prefiller-ports ${ENGINE_PORT} \
  --decoder-hosts ${DECODE_IP} \
  --decoder-ports ${ENGINE_PORT}

# === 验证请求 (proxy 就绪后执行) ===
# curl -sS -X POST http://192.168.13.198:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
#     "model": "hy4_w8a8_v2",
#     "messages": [{"role": "user", "content": "Who are you?"}],
#     "max_tokens": 256,
#     "temperature": 0
# }'
#
# 注意: prompt tokens + max_tokens 必须 <= P/D 节点的 MAX_MODEL_LEN (512).
#       proxy 发往 P 前会把 max_tokens 改写为 1 (仅 prefill), 发往 D 的仍是原始 max_tokens,
#       超限时 P 正常而 D 返回 400, 480s 后 P 报 "Force freed expired request".
#
# 健康检查 (返回 prefiller/decoder 实例数):
# curl http://192.168.13.198:8000/healthcheck
```
