FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 备份原始源文件 (可选，但推荐)
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak

# 写入新的源
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.sustech.edu.cn@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.sustech.edu.cn@g" /etc/apt/sources.list

# 安装常用的开发工具和依赖项
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    nano \
    locales \
    python3-opengl \
    libfreetype6-dev \
    libfontconfig1-dev \
    libegl1 libgl1 libgl1-mesa-glx libgles2 \
    && rm -rf /var/lib/apt/lists/*

# 设置区域设置
RUN sed -i '/en_US.UTF-8 UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

# 创建一个非 root 用户用于开发
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 设置工作目录
WORKDIR /home/$USERNAME/workspace

# 注意: 本项目在 rootless Docker 下以 root 用户运行(容器 root 映射到宿主用户),
# 因此通用依赖必须安装到系统 site-packages。若在 USER vscode 之后安装, 包会落到
# /home/vscode/.local, 而容器内的 root 看不到它们 (tensorboard/wandb/moviepy 等)。
# 清华 PyPI 镜像当前会返回 403, 使用与 apt 一致的南科大镜像。
RUN pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/simple/ && \
    pip install gymnasium wandb moviepy==1.0.3 tensorboard imageio requests matplotlib \
        -i https://mirrors.sustech.edu.cn/pypi/simple/

# 切换到非 root 用户 (devcontainer 默认用户)
USER $USERNAME

# 注意: mamba-ssm 需要现场编译 CUDA 扩展，构建非常慢且容易失败。
# 当前训练使用的是 Attention + GRU 模型，且 benchmarl/models/mamba.py 对 mamba_ssm 是延迟导入，
# 因此默认跳过安装，不影响训练。若确实要用 Mamba 模型，请手动取消下面一行的注释。
# RUN pip install mamba-ssm[causal-conv1d] --no-build-isolation 