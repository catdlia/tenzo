FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Додаємо всі необхідні бібліотеки розробки
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    gnupg \
    software-properties-common \
    zlib1g-dev \
    libzstd-dev \
    libedit-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libffi-dev \
    gdb \
    libomp-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Встановлюємо LLVM/MLIR 18
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 all && \
    rm llvm.sh

# 3. Довстановлюємо специфічні MLIR пакети
RUN apt-get update && apt-get install -y \
    libmlir-18-dev \
    mlir-18-tools \
    && rm -rf /var/lib/apt/lists/*

# 4. Встановлюємо Vulkan SDK та Intel GPU drivers
RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && \
    apt-get update && apt-get install -y \
    vulkan-sdk \
    libvulkan-dev \
    vulkan-tools \
    spirv-tools \
    mesa-vulkan-drivers \
    libgl1-mesa-dri \
    intel-media-va-driver \
    vainfo \
    && rm -rf /var/lib/apt/lists/*

# 5. Встановлюємо інструменти для прискорення збірки
RUN apt-get update && apt-get install -y lld ccache && rm -rf /var/lib/apt/lists/*

ENV LLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
ENV MLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir
ENV PATH="/usr/lib/llvm-18/bin:${PATH}"
ENV VULKAN_SDK=/usr

WORKDIR /app

CMD ["/bin/bash"]