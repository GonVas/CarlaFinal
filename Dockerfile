FROM nvidia/vulkan:1.1.121-cuda-10.1-alpha

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \

    curl \

    ca-certificates \

    sudo \

    git \

    bzip2 \

    wget \

    build-essential \ 

    cmake \ 

    unzip \

    pkg-config \

    libjpeg-dev \ 

    libpng-dev \

    libtiff-dev \

    python-opencv \

    python3-pip \

    libx11-6 \

 && rm -rf /var/lib/apt/lists/*


RUN apt-get install python-opencv


RUN apt-get -y update
RUN apt-get -y upgrade 

RUN apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig


RUN packages='libsdl2-2.0' \
    && apt-get update && apt-get install -y $packages --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*



RUN mkdir /project
RUN mkdir /project/carla

WORKDIR /project/carla

COPY . /project/carla

# Create a non-root user and switch to it

RUN adduser --disabled-password --gecos '' --shell /bin/bash user \

 && chown -R user:user /project/carla

RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

RUN chown -R user:user /home/user

RUN chown -R user:user /project/


USER user


# All users can use /home/user as their home directory

ENV HOME=/home/user

RUN chmod 777 /home/user

#COPY cnn_vae.py /project/carla

#COPY util.py /project/carla

#COPY carla_data_small/ /project/carla/carla_data_small

#COPY . /project/carla


RUN mkdir recons

RUN mkdir trained_models_ckpts



#COPY CARLA_0.9.9.tar.gz /project/carla

RUN tar -xzvf CARLA_0.9.9.tar.gz




# Install Miniconda and Python 3.8

ENV CONDA_AUTO_UPDATE_CONDA=false

ENV PATH=/home/user/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \

 && chmod +x ~/miniconda.sh \

 && ~/miniconda.sh -b -p ~/miniconda \

 && rm ~/miniconda.sh \

 && conda install -y python==3.8.1 \

 && conda clean -ya



# CUDA 10.2-specific steps

RUN conda install -y -c pytorch \

    pip \


    cudatoolkit=10.2 \

    "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \

    "torchvision=0.6.0=py38_cu102" \

 && conda clean -ya


RUN pip install --user gdown

RUN pip install --user gym

RUN pip install --user networkx


ENV SDL_VIDEODRIVER=offscreen


RUN python3 -m easy_install PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg; exit 0


#CMD sh ./CarlaUE4.sh & sleep 15 && python3 main.py --production

CMD /bin/bash