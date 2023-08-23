#!/bin/bash

#
# Script to initialize a new container based on the deepar docker
#

container_name="next_${USER}"
image_name="next"

cmd="docker run -d -it --rm --name ${container_name} --entrypoint bash ${image_name}"

echo $cmd
$cmd

echo "Your container is '$container_name', running image '$image_name'"
