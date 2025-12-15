FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

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

# 切换到非 root 用户
USER $USERNAME

# 设置工作目录
WORKDIR /home/$USERNAME/workspace
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install wandb moviepy==1.0.3 tensorboard imageio requests matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple