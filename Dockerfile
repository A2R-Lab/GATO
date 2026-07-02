# Optional container build — the primary install path is HOST-NATIVE
# (./tools/install.sh + ./tools/build.sh, see README). Use this only for a
# reproducible environment; it just wraps the same install script.
#
#   docker build -t gato .
#   docker run --gpus all -it -v $PWD:/workspace gato
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        python3 python3-dev python3-venv python3-pip \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
# auto-activate the project venv created by tools/install.sh
RUN echo "[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate" >> /root/.bashrc

CMD ["/bin/bash"]
