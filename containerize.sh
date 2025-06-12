#!/bin/bash

image_name=$1

sudo docker compose -f docker-compose.yolo.yaml down
sudo docker compose -f docker-compose.yolo.yaml up -d

prefix="${image_name%%:*}:"

sudo docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | \
grep "^$prefix" | \
grep -v "$image_name" | \
awk '{print $2}' | xargs -r sudo docker rmi


