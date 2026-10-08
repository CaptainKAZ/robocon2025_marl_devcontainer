#!/usr/bin/env bash
# 重建训练容器 robocon2025-marl（特权模式）。
#
# 背景/约定（用户 2026-10-03 要求）：以后起这个容器都用 --privileged。
#   - 特权模式下容器内 root 拥有 CAP_SYS_PTRACE，才能对训练进程做 ptrace/gdb 注入（热挂观察窗等）；
#     非特权容器 + host yama.ptrace_scope=1 时完全无法附着。
#   - 注意：创建参数（--privileged 等）只在 docker run 时生效；`docker start` 无法追加，
#     所以要让特权生效必须重建容器（docker rm -f + docker run）。
#   - `-u root`：镜像默认 USER 是 vscode（uid 1000），在 rootless docker 下映射到 host 的
#     子 uid（100000+），对 bind mount 的仓库**只读** ⇒ 必须用 root（映射 host proton 1000）。
#     否则训练写 outputs/ 会 Permission denied。
#
# 用法：bash .opencode/start_container.sh          # 重建（有训练在跑会拒绝执行）
#       FORCE=1 bash .opencode/start_container.sh  # 强行重建（会中断训练）
set -euo pipefail

NAME=robocon2025-marl
HOST=/home/proton/robocon2025_marl_devcontainer
IMG=robocon2025-marl:latest

# 只把"comm 是 python 且命令行含 clear_restore.py"的进程算作训练在跑；
# 否则 ps 的 argv 里只要出现该文件名（例如调用方 shell 的命令行）就会误判。
if ps -eo comm,args | awk '$1 ~ /^python/ && $0 ~ /clear_[r]estore\.py/ {found=1} END{exit !found}' && [ "${FORCE:-0}" != "1" ]; then
  echo "!! 检测到训练正在运行（clear_restore.py）；先停训练，或用 FORCE=1 强行重建" >&2
  exit 1
fi

echo ">> 删除旧容器 $NAME（若有）"
docker rm -f "$NAME" 2>/dev/null || true

echo ">> 以特权模式创建 $NAME"
docker run -d --name "$NAME" \
  --privileged \
  -u root \
  --gpus=all --shm-size=8gb \
  -v "$HOST":/home/vscode/workspace \
  -e PIP_INDEX_URL=https://mirrors.sustech.edu.cn/pypi/simple/ \
  "$IMG" sleep infinity

docker ps --filter "name=$NAME" --format '{{.Names}} | {{.Status}}'

# 项目依赖自愈（2026-10-03 教训）：这些依赖原先只存在于容器可写层，`docker rm -f` 重建后会丢失，
# 表现为启动训练立刻 ModuleNotFoundError（imageio 缺失则只是 eval worker 报错，训练照跑）。
# Dockerfile 现已声明通用依赖，但为兼容旧镜像，这里仍做幂等检查 + 自动补装：
#   ① 通用依赖（与 .devcontainer/Dockerfile 的 pip 列表保持一致）
#   ② 本地源码包 editable 安装（指向 bind mount 的源码，重建后代码依然最新）
GEN_DEPS=(gymnasium wandb moviepy==1.0.3 tensorboard imageio requests matplotlib
          "torchrl~=0.8.0" tqdm hydra-core "av<14"
          "pyglet<=1.5.27" gym six "gym-notices==0.0.8")
if ! docker exec "$NAME" python -c 'import torchrl, tensordict, hydra, av, imageio, matplotlib, gymnasium, moviepy, tensorboard, wandb, pyglet, gym' >/dev/null 2>&1; then
  echo ">> 检测到通用依赖缺失，开始补装（与 Dockerfile 一致）..."
  docker exec "$NAME" pip install "${GEN_DEPS[@]}" \
    || { echo "!! 通用依赖补装失败，请手动执行上面的 pip install" >&2; exit 1; }
  echo ">> 通用依赖补装完成"
else
  echo ">> 通用依赖检查：torchrl/tensordict/hydra/av/imageio/matplotlib/gymnasium/moviepy/tensorboard/wandb/pyglet/gym 均在 ✓"
fi

if ! docker exec "$NAME" python -c 'import vmas, benchmarl' >/dev/null 2>&1; then
  echo ">> 检测到本地源码包缺失，开始 editable 补装（benchmarl/vmas）..."
  docker exec "$NAME" pip install -e /home/vscode/workspace/BenchMARL -e /home/vscode/workspace/VectorizedMultiAgentSimulator \
    || { echo "!! 本地源码包补装失败，请手动执行上面的 pip install" >&2; exit 1; }
  echo ">> 本地源码包补装完成"
else
  echo ">> 本地源码包检查：vmas / benchmarl 均在 ✓"
fi

echo ">> Privileged = $(docker inspect "$NAME" --format '{{.HostConfig.Privileged}}')"
echo ">> 之后服务器重启只需：docker start $NAME"
