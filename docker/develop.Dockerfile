FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS cuda

FROM tay2510/fake_sentinel:base

COPY --from=cuda /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=cuda /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=cuda /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg

# GPU-related Packages
RUN pip --no-cache-dir install --upgrade \
    # Deep Learning
    torch \
    torchvision \
    ray[tune] \
    # -------------------------
    # Utilities
    glances[gpu]

# Testing Time Packages for DFDC Competition
COPY libraries/*.whl /tmp/

RUN pip --no-cache-dir install /tmp/*.whl
