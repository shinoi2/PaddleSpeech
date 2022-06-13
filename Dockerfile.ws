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
    ln -s /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcufft.so.10 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcufft.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so

RUN python3 -m pip install pip --upgrade && \
    pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    pip install paddlespeech_ctcdecoders protobuf==3.20.1 pytest-runner

RUN wget https://paddlespeech.bj.bcebos.com/Parakeet/nltk_data.tar.gz \
    -O /root/nltk_data.tar.gz && \
    tar -xvf /root/nltk_data.tar.gz -C /root/ && \
    wget https://paddlespeech.bj.bcebos.com/s2t/aishell/asr0/asr0_deepspeech2_online_aishell_fbank161_ckpt_1.0.1.model.tar.gz \
    --directory-prefix /root/.paddlespeech/models/deepspeech2online_aishell-zh-16k/1.0/ && \
    wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm \
    --directory-prefix /root/.paddlespeech/models/language_model/data/lm/

ADD . /root/PaddleSpeech

EXPOSE 8091

RUN cd /root/PaddleSpeech && pip install .

ENTRYPOINT cd /root/PaddleSpeech/paddlespeech/server/ && paddlespeech_server start --config_file ./conf/ws_application.yaml
