FROM nvcr.io/nvidia/pytorch:19.12-py3

WORKDIR "/workspace"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y 
RUN apt-get install -y vim git  ffmpeg libportaudio2 cmake python3.6-dev python3.7 python3.7-dev \
	ffmpeg libsndfile1 lame cmake openssh-server python3-pyqt5 xauth python3-pip \
	pulseaudio osspd-pulseaudio pulseaudio-module-jack gstreamer1.0-pulseaudio \
	pulseaudio-dlna pulseaudio-esound-compat libgstreamer-plugins-base1.0-0 \
	libgstreamer-plugins-good1.0-0 libgstreamer-plugins-bad1.0-0 liballegro-acodec5.2 libavcodec57 \
	python-translitcodec xmms2-plugin-avcodec xubuntu-restricted-extras python-soundfile \
	libsndifsdl2-dev python3-soundfile python-soundfile-doc \
        && apt-get -y autoremove

RUN python -m pip install matplotlib
RUN python -m pip install inflect
RUN python -m pip install librosa
RUN python -m pip install scipy
RUN python -m pip install tensorboardX
RUN python -m pip install Unidecode
RUN python -m pip install pillow
RUN python -m pip install nltk
RUN python -m pip install jamo
RUN python -m pip install music21
RUN python -m pip install pydub starlette uvicorn ujson aiofiles

RUN conda install tensorflow\<2.0.0

RUN apt-get install -y musescore

RUN rm -Rf /root/.cache/*

RUN apt-get clean

RUN systemd-tmpfiles --create

RUN mkdir /run/sshd

RUN mkdir ./logs
RUN chmod 777 ./logs

RUN touch ./add_user.sh
RUN echo 'nohup tensorboard --logdir /workspace/logs --host 0.0.0.0 &' >> add_user.sh
RUN echo 'adduser --quiet --disabled-password --gecos "$1,,," --uid $2 $1' >> add_user.sh
RUN echo 'echo "export PATH=/usr/local/nvm/versions/node/v13.3.0/bin:\$PATH" >> /home/${1}/.bashrc' >> add_user.sh
RUN echo 'echo "export PATH=/opt/conda/bin:/opt/cmake-3.14.6-Linux-x86_64/bin/:\$PATH" >> /home/${1}/.bashrc' >> add_user.sh
RUN echo 'echo "export PATH=/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH" >> /home/${1}/.bashrc' >> add_user.sh
RUN echo 'echo Added user ${1} ${2}' >> add_user.sh
RUN echo 'login -f $1' >> add_user.sh
RUN chmod +x add_user.sh
RUN sed -i 's/"\$\@"/\$\@/' /usr/local/bin/nvidia_entrypoint.sh

WORKDIR /
COPY run_mellotron.py /
COPY train_utils.py /
COPY checkpoints /checkpoints 
COPY mellotron /mellotron 
COPY musicXML /musicXML

ENTRYPOINT ["python", "-X", "utf8", "run_mellotron.py"]