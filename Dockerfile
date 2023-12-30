FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# basic tools
WORKDIR /home
RUN apt update \
  && apt install -y --no-install-recommends \
  git vim openssh-client gnupg curl wget ca-certificates unzip zip less zlib1g sudo coreutils sed grep \
  pkg-config libssl-dev nvidia-container-toolkit

# cargo/rust
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# https://blog.rust-lang.org/2022/06/22/sparse-registry-testing.html
ENV CARGO_UNSTABLE_SPARSE_REGISTRY=true
RUN set -eux; \
  apt update \
  && apt install -y --no-install-recommends \
    ca-certificates gcc build-essential; \
  url="https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"; \
  wget "$url"; \
  chmod +x rustup-init; \
  ./rustup-init -y --no-modify-path --default-toolchain nightly; \
  rm rustup-init; \
  chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
  rustup --version; \
  cargo --version; \
  rustc --version;
#

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
RUN echo "export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}" >> ~/.bashrc
RUN export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}

