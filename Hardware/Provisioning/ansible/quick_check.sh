#!/bin/bash
echo "Quick health check on all nodes..."

ansible all -i inventory.yml -m shell -a "hostname && nvidia-smi -L | wc -l && ibstat | grep -i state"
ansible all -i inventory.yml -m shell -a "dcgmi discovery -l"
ansible gpu_nodes -i inventory.yml -m shell -a "nvbandwidth -t" --one-line
ansible gpu_nodes -i inventory.yml -m shell -a "docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi"
