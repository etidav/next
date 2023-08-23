#!/bin/bash

#
# Script to enter a previously started container deepar
#

container_name="next_${USER}"

IP_DOCKER=$(docker ps | grep -w $container_name | cut -d' ' -f1)
echo "The unique ID of your docker is: $IP_DOCKER"
cmd="docker exec -it ${IP_DOCKER} bash"

echo "will now execute:"
echo $cmd
$cmd

true
