#!/bin/bash

#
# Script to stop a previously started container deepar

container_name="next_${USER}"

cmd="docker rm -f ${container_name}"

echo "will now execute:"
echo $cmd
$cmd

true
