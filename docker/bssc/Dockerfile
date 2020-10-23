FROM nvidia/cudagl:10.1-devel-ubuntu18.04

# Arguments
ARG user=pytorch
ARG uid=1000
ARG gid=1000
ARG gpu_arch=sm_70

ENV DEBIAN_FRONTEND noninteractive
#Assume major version 3.
ENV PYTHON_VERSION 3.6

# Install some dependencies
RUN apt-get update && apt-get install -y \
		wget\
    git\
		"python$PYTHON_VERSION"\
    python3-pip\
    sudo\
		vim\
		unzip\
		libgtk2.0\
		libglu1-mesa-dev\
		freeglut3-dev\
		mesa-common-dev\
		libosmesa6-dev\
		libegl1-mesa-dev\
		libglvnd-dev\
		libgl1-mesa-dev\
		libgles2-mesa-dev\
		cmake\
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

#GLVND setup
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

#Install VTK with EGL support
# RUN git clone https://gitlab.kitware.com/vtk/vtk.git --branch nightly-master
# RUN mkdir /vtk/build && cd /vtk/build && \
#     cmake \
# 		-DVTK_OPENGL_HAS_EGL:BOOL=ON\
# 		-DVTK_USE_X:BOOL=OFF\
# 		-DVTK_BUILD_SHARED_LIBS:BOOL=OFF\
# 		-DBUILD_TESTING=OFF\
# 		-DVTK_BUILD_EXAMPLES=OFF\
# 		-DCMAKE_BUILD_TYPE=Release\
# 		-DVTK_WRAP_PYTHON=ON\
# 		-DVTK_PYTHON_VERSION=3\
# 		-DVTK_PYTHON_SITE_PACKAGES_SUFFIX="lib/python$PYTHON_VERSION/dist-packages"\
# 		.. &&\
# 		make install -j"$(nproc)"

#Install VTK with EGL support
RUN git clone https://gitlab.kitware.com/vtk/vtk.git --branch v8.2.0
COPY vtk_setup.py /vtk/build/setup.py
RUN cd /vtk/build && \
    cmake \
		-DVTK_OPENGL_HAS_EGL:BOOL=ON\
		-DVTK_USE_X:BOOL=OFF\
		-DVTK_BUILD_SHARED_LIBS:BOOL=OFF\
		-DBUILD_TESTING=OFF\
		-DBUILD_EXAMPLES=OFF\
		-DCMAKE_BUILD_TYPE=Release\
		-DVTK_WRAP_PYTHON=ON\
		-DVTK_PYTHON_VERSION=3\
		-DVTK_PYTHON_SITE_PACKAGES_SUFFIX="python$PYTHON_VERSION/dist-packages"\
		.. &&\
		make install -j"$(nproc)" &&\
		ldconfig &&\
		python3 setup.py install

#Install repo dependencies
RUN pip3 install --no-cache-dir\
		opencv-python-headless\
		torch \
		torchvision \
		tb-nightly \
		future \
		Cython \
		matplotlib \
		numpy \
		scipy \
		pyyaml \
		scikit-image \
		gitpython \
		sklearn \
    moviepy \
		seaborn\
		mayavi

# Setup user
RUN export uid="${uid}" gid="${gid}" && \
    groupadd -g "${gid}" "${user}" && \
    useradd -m -u "${uid}" -g "${user}" -s /bin/bash "${user}" && \
    passwd -d "${user}" && \
    usermod -aG sudo "${user}"

# Switch to user
USER "${uid}"

CMD bash

VOLUME data
VOLUME host_home
