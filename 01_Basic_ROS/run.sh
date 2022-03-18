#!/bin/bash
cd "$(dirname "$0")"

xhost +

docker build -t mcv_mai/motion_recognition -f ./Dockerfile . && \
docker run --rm -it \
    --privileged \
    --device /dev/video0 \
    -e QT_X11_NO_MITSHM=1 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v `pwd`/src:/src \
    mcv_mai/motion_recognition:latest \
    bash -c "cd /src && \
    colcon build && \
    source /src/install/setup.bash && \
    ros2 launch motion_recognition motion_recognition.launch.py"
