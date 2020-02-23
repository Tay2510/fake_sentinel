FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS cuda

FROM tay2510/fake_sentinel:base

COPY --from=cuda /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=cuda /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=cuda /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg

# CUDA
ENV CUDA_VERSION=10.0.130
ENV CUDA_PKG_VERSION=10-0=$CUDA_VERSION-1
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=10.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-cupti-$CUDA_PKG_VERSION \
      cuda-cudart-$CUDA_PKG_VERSION \
      cuda-cudart-dev-$CUDA_PKG_VERSION \
      cuda-libraries-$CUDA_PKG_VERSION \
      cuda-libraries-dev-$CUDA_PKG_VERSION \
      cuda-nvml-dev-$CUDA_PKG_VERSION \
      cuda-minimal-build-$CUDA_PKG_VERSION \
      cuda-command-line-tools-$CUDA_PKG_VERSION \
      libcudnn7=7.5.0.56-1+cuda10.0 \
      libcudnn7-dev=7.5.0.56-1+cuda10.0 \
      libnccl2=2.4.2-1+cuda10.0 \
      libnccl-dev=2.4.2-1+cuda10.0 && \
    ln -s /usr/local/cuda-10.0 /usr/local/cuda

# GPU-related Packages
RUN pip --no-cache-dir install --upgrade \
    # Deep Learning
    torch \
    torchvision \
    torchsummary \
    pretrainedmodels \
    # -------------------------
    # Data Augmentation
    imgaug \
    kornia \
    # -------------------------
    # Hyperparemeter Tuning
    ray[tune] \
    hyperopt \
    bayesian-optimization \
    nevergrad \
    scikit-optimize \
    # -------------------------
    # Utilities
    glances[gpu]

# Testing Time Packages for DFDC Competition
COPY libraries/*.whl /tmp/

RUN pip --no-cache-dir install /tmp/*.whl
