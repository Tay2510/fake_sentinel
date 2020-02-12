FROM python:3.6.10-slim

# System
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    git \
    cmake \
    curl \
    vim \
    libgtk2.0-dev

# Common Packages
RUN pip --no-cache-dir install --upgrade \
    # -------------------------
    # Data Science
    numpy \
    scipy \
    sklearn \
    pandas \
    # -------------------------
    # Image Processing
    opencv-contrib-python \
    scikit-image \
    pillow \
    imageio \
    imageio-ffmpeg \
    # -------------------------
    # Visualization
    matplotlib \
    seaborn \
    plotly \
    # -------------------------
    # Jupyter Notebook
    ipykernel \
    jupyter \
    jupyterthemes \
    jupyter_contrib_nbextensions \
    jupyterlab
