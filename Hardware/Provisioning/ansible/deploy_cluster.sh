#!/bin/bash
# Run ONCE from your laptop or bastion (not on master yet)
set -e

echo "Deploying 4-node HPC/AI cluster..."

# Generate fresh SSH key if not exists
[ ! -f ~/.ssh/id_rsa ] && ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa

# Install Ansible + sshpass if missing
sudo apt update && sudo apt install -y ansible sshpass

# Copy SSH key to all nodes (including master)
for ip in 172.31.36.{195..198}; do
  echo "Pushing SSH key to $ip..."
  sshpass -p ubuntu ssh-copy-id -o StrictHostKeyChecking=no ubuntu@$ip
done

# Run full playbook
ansible-playbook -i inventory.yml full_deploy.yml --extra-vars "cluster_master_ip=172.31.36.195"

echo "CLUSTER READY! Run quick checks:"
echo "  ssh master"
echo "  ansible all -i inventory.yml -m shell -a 'nvidia-smi topo -m'"
echo "  ansible all -i inventory.yml -m shell -a 'dcgmi discovery -l'"
