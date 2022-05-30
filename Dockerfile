FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64" CUDA_VISIBLE_DEVICES=0
RUN apt update && \
    apt install -y \
        build-essential \
        git \
        libsndfile1 \
        python3-pip \
        wget && \
    apt clean autoclean && \
    apt autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so.11 ./usr/local/cuda-11.2/targets/x86_64-linux/lib/libcublas.so && \
    ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so.11 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcusolver.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so

RUN pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    pip install pytest-runner && \
    pip install .

RUN mkdir -p /root/.paddlespeech/models/pwgan_csmsc-zh/ && wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip -O pwg_baker_ckpt_0.4.zip && \
    mkdir -p /root/.paddlespeech/models/fastspeech2_csmsc-zh/ && wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip -O fastspeech2_nosil_baker_ckpt_0.4.zip && \
    mkdir -p /root/.paddlespeech/models/conformer_wenetspeech-zh-16k/ && wget https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar.gz -O asr1_conformer_wenetspeech_ckpt_0.1.1.model.tar && \
    mkidr -p /root/.paddlespeech/models/panns_cnn14-32k/ && wget https://paddlespeech.bj.bcebos.com/cls/panns_cnn14.tar.gz -O panns_cnn14.tar.gz

ADD . /root/PaddleSpeech

ENTRYPOINT cd /root/PaddleSpeech/paddlespeech/server && paddlespeech_server start --config_file ./conf/application.yaml
