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
    ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcufftw.so.10 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcufftw.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so

RUN python3 -m pip install pip --upgrade && \
    pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    pip install paddlespeech_ctcdecoders protobuf==3.20.1 pytest-runner

RUN wget https://paddlespeech.bj.bcebos.com/Parakeet/nltk_data.tar.gz \
    -O /root/nltk_data.tar.gz && \
    tar -xvf /root/nltk_data.tar.gz -C /root/ && \
    mkdir -p /root/.paddlespeech/models/deepspeech2offline_aishell-zh-16k/1.0/ && \
    wget https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz \
    -O /root/.paddlespeech/models/deepspeech2offline_aishell-zh-16k/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz && \
    ln -s /root/.paddlespeech/models/deepspeech2offline_aishell-zh-16k/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz \
    /root/.paddlespeech/models/deepspeech2offline_aishell-zh-16k/1.0/asr0_deepspeech2_aishell_ckpt_0.1.1.model.tar.gz && \
    mkdir -p /root/.paddlespeech/models/language_model/data/lm/1.0/ && \
    wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm \
    -O /root/.paddlespeech/models/language_model/data/lm/zh_giga.no_cna_cmn.prune01244.klm && \
    ln -s /root/.paddlespeech/models/language_model/data/lm/zh_giga.no_cna_cmn.prune01244.klm \
    /root/.paddlespeech/models/language_model/data/lm/1.0/zh_giga.no_cna_cmn.prune01244.klm && \
    mkdir -p /root/.paddlespeech/models/fastspeech2_csmsc-zh/1.0/ && \
    wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_static_0.4.zip \
    -O /root/.paddlespeech/models/fastspeech2_csmsc-zh/fastspeech2_nosil_baker_static_0.4.zip && \
    ln -s /root/.paddlespeech/models/fastspeech2_csmsc-zh/fastspeech2_nosil_baker_static_0.4.zip \
    /root/.paddlespeech/models/fastspeech2_csmsc-zh/1.0/fastspeech2_nosil_baker_static_0.4.zip && \
    mkdir -p /root/.paddlespeech/models/pwgan_csmsc-zh/1.0/ && \
    wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_static_0.4.zip \
    -O /root/.paddlespeech/models/pwgan_csmsc-zh/pwg_baker_static_0.4.zip && \
    ln -s /root/.paddlespeech/models/pwgan_csmsc-zh/pwg_baker_static_0.4.zip \
    /root/.paddlespeech/models/pwgan_csmsc-zh/1.0/pwg_baker_static_0.4.zip

ADD . /root/PaddleSpeech

EXPOSE 8090

RUN cd /root/PaddleSpeech && pip install .

ENTRYPOINT cd /root/PaddleSpeech/paddlespeech/server/ && paddlespeech_server start --config_file ./conf/application.yaml
