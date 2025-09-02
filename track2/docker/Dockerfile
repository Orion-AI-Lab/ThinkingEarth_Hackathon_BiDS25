FROM nvcr.io/nvidia/dgl:24.09-py3 
#FROM nvcr.io/nvidia/pytorch:25.05-py3

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade setuptools
RUN python -m pip install --no-cache-dir --upgrade wheel

# The custom CUDA extension does not suppport architerctures < 7.0
ENV FORCE_CUDA_EXTENSION=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 9.0+PTX"

# torch-harmonics, exclude the version to make it work (before ==0.7.1)
#RUN pip install --no-build-isolation torch-harmonics==0.7.4

RUN --mount=type=secret,id=netrc,target=/root/.netrc,required=true \
    python -m pip install --no-cache-dir \
        --requirement /tmp/requirements.txt \
        # DVL group package index
        --extra-index-url https://gitlab-master.nvidia.com/api/v4/groups/106305/-/packages/pypi/simple \
        # NCoreSDK package index
        --extra-index-url https://gitlab-master.nvidia.com/api/v4/projects/61004/packages/pypi/simple \
        #--find-links https://download.pytorch.org/whl/torch \
        #--find-links https://download.pytorch.org/whl/torchvision \
    && rm /tmp/requirements.txt
     
ENV DEBIAN_FRONTEND=
WORKDIR /bids2025_weather