#!/bin/bash

docker run --rm -it \
	mcv_mai/motion_recognition:latest \
	bash -c "ros2 topic echo /motion_flag"


