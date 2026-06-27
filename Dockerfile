FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    python3-dev \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libosmesa6 \
    libosmesa6-dev \
    libglfw3 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libdbus-1-3 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-randr0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/balance-robot-mujoco-sim

COPY requirements.txt* ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install mujoco numpy scipy matplotlib PySide6

CMD ["/bin/bash"]
