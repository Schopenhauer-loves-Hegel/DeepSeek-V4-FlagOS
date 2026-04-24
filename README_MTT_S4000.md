# DeepSeek-V4 FP8 推理 (摩尔线程 MTT S4000, 8机64卡)

## 1. 下载权重

从 HuggingFace 下载 DeepSeek-V4 原始权重：

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4 --local-dir /path/to/DeepSeek-V4-HF
```

## 2. 切分权重 (MP=64)

已切分好的 FP8 MP64 权重位于 MR-gpu50-new (10.7.66.194)：

```
/public-nfs/tj/0423/model/DSV4-fp8-mp64
```

如需自行切分，使用多进程并行切分，`--num-workers` 控制并发数（默认等于 MP，内存不够可调小）：

```bash
python convert_0424_try_mp.py \
    --hf-ckpt-path /path/to/DeepSeek-V4-HF \
    --save-path /path/to/DeepSeek-V4-HF-FP8-MP64 \
    --n-experts 384 \
    --model-parallel 64 \
    --o-groups 16 \
    --num-workers 8
```

## 3. 拉取镜像

```bash
docker pull harbor.baai.ac.cn/flagos-inner-models-release/lagrelease-mthreads-deepseek-v4-pro:202604242342
```

## 4. 启动容器

在每个节点上执行：

```bash
docker run -itd --privileged --net host --name=flagos \
    -w /workspace \
    -v /public-nfs/:/public-nfs/ \
    -v /public/:/public/ \
    --env MTHREADS_VISIBLE_DEVICES=all \
    --shm-size=80g \
    harbor.baai.ac.cn/flagos-inner-models-release/lagrelease-mthreads-deepseek-v4-pro:202604242342 \
    /bin/bash
```

## 5. 启动推理

在每个节点的容器中执行：

```bash
cd /workspace/code/
bash run_node.sh
```
