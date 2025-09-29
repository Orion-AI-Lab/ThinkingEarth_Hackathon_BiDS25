ARG TAG=0.0.0-dev
FROM bids2025_weather_base:${TAG}

#########################
# Devcontainer user setup
#########################

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git-lfs \
    nano \
    tree \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.4/files/ngccli_linux.zip \
    -O ngccli_linux.zip \
    && unzip ngccli_linux.zip \
    && rm ngccli_linux.zip
RUN chmod +x ngc-cli/ngc
ENV PATH="/opt/ngc-cli:${PATH}"

# Ensures file permissions are consistent
ARG USER_ID
ARG GROUP_ID
RUN groupadd --gid $GROUP_ID user || echo "Group already exists"
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# Allow passwordless sudo in case the user needs to install something on the fly
RUN apt-get update && apt-get install -y \
    sudo \
    && rm -rf /var/lib/apt/lists/*
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Pretty terminal colors
ENV TERM=xterm-256color

# So the user is not faced with an empty list of apt packages
RUN apt-get update

USER user

# Bash Fuzzy Finder
RUN git clone --depth 1 https://github.com/junegunn/fzf.git --branch 0.50.0 $HOME/.fzf
RUN $HOME/.fzf/install

# Set this env variable to make the distributed trainings run otherwise they get stuck
ENV NCCL_P2P_DISABLE=1 
#(in terminal one can put "export NCCL_P2P_DISABLE=1", is this equivalent to this command?)

ENV DEBIAN_FRONTEND=
WORKDIR /bids2025_weather