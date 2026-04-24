# DeepSeek-V4 FP8 推理 (摩尔线程 MTT S4000, 8机64卡)

## 1. 下载权重

从 HuggingFace 下载 DeepSeek-V4 模型权重到本地共享存储：

```bash
# 需要安装 huggingface-cli
huggingface-cli download deepseek-ai/DeepSeek-V4 --local-dir /path/to/DeepSeek-V4-HF
```

## 2. 切分权重 (MP=64)

使用多进程并行切分，`--num-workers` 控制并发数（默认等于 MP，内存不够可调小）：

```bash
python convert_0424_try_mp.py \
    --hf-ckpt-path /path/to/DeepSeek-V4-HF \
    --save-path /path/to/DeepSeek-V4-HF-FP8-MP64 \
    --n-experts 256 \
    --model-parallel 64 \
    --o-groups 8 \
    --num-workers 8
```

## 3. 启动推理

8 机 64 卡，每个节点 8 卡：

```bash
torchrun \
    --nnodes=8 \
    --nproc_per_node=8 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    generate_new_encoding.py \
        --ckpt-path /path/to/DeepSeek-V4-HF-FP8-MP64 \
        --config config.json \
        --interactive \
        --max-new-tokens 300
```
