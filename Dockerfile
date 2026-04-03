FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install -y torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r /workspace/requirements.txt

CMD ["/bin/bash"]