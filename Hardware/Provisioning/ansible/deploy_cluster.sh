#!/bin/bash
# Run ONCE from your laptop or bastion (not on master yet)
set -e

echo "Deploying 4-node HPC/AI cluster..."

# Install Ansible + sshpass if missing
sudo apt update && sudo apt install -y ansible sshpass

# Run full playbook
ansible-playbook -i inventory.yml full_deploy.yml --extra-vars "cluster_master_ip=172.31.36.195"

echo "CLUSTER READY! Run quick checks:"
echo "  ssh master"
echo "  ansible all -i inventory.yml -m shell -a 'nvidia-smi topo -m'"
echo "  ansible all -i inventory.yml -m shell -a 'dcgmi discovery -l'"
