#running command in the ansible folder

ansible all -i inventory/inventory.yml quick_check.yml --check # quick_check.yml include the quick_check.sh script

ansible-playbook -i inventory/inventory.yml deploy-nvidia.yml --check

#ansible directory structure 
ansible
--> inventory/ deploy-nvidia.yml full_deploy.yml  
inventory/
+-- hosts.yml
+-- group_vars/
�   +-- all.yml





### Cluster Overview
```
master   → OS: 172.31.36.195   BMC: 172.31.53.155
client01 → OS: 172.31.36.196   BMC: 172.31.53.156
client02 → OS: 172.31.36.197   BMC: 172.31.53.157
client03 → OS: 172.31.36.198   BMC: 172.31.53.158
```
### Full Automation Suite (4 Scripts + Inventory)

#### 1. `inventory.yml` – Ansible inventory (save as-is)
```yaml
all:
  hosts:
    master:
      ansible_host: 172.31.36.195
    client01:
      ansible_host: 172.31.36.196
    client02:
      ansible_host: 172.31.36.197
    client03:
      ansible_host: 172.31.36.198
  children:
    clients:
      hosts:
        client01:
        client02:
        client03:
    gpu_nodes:
      hosts:
        client01:
        client02:
        client03:
```

#### 2. `deploy_cluster.sh` – One-command full deployment
```bash
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
```

#### 3. `full_deploy.yml` – Main Ansible playbook
```yaml
---
- name: Bootstrap & configure full 4-node HPC cluster
  hosts: all
  become: yes
  vars:
    cluster_master_ip: "{{ cluster_master_ip }}"
  tasks:
    - name: Set hostname
      hostname:
        name: "{{ inventory_hostname }}"

    - name: Update /etc/hosts
      lineinfile:
        path: /etc/hosts
        line: "{{ hostvars[item]['ansible_host'] }} {{ item }}"
      loop: [master, client01, client02, client03]

    - name: Install essential tools
      apt:
        name:
          - sshpass, ipmitool, dos2unix, make, gcc, hwloc, numactl, net-tools
          - pv, nload, iftop, unzip, dkms, jq, nfs-common, openmpi-bin
          - cifs-utils, nmon, policycoreutils-python-utils, python3-pip
          - python3-matplotlib, stress-ng, fio, sysstat, nvme-cli, infiniband-diags
          - libibumad3, mstflint, powertop
        state: present
        update_cache: yes

    - name: Add NVIDIA CUDA repo
      shell: |
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb
      args:
        creates: /etc/apt/sources.list.d/cuda-*.list

    - name: Add Docker repo
      shell: |
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.asc
      args:
        creates: /etc/apt/sources.list.d/docker.list

    - name: Add NVIDIA Container Toolkit repo
      shell: |
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
          sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sed -i '/experimental/s/^#//' /etc/apt/sources.list.d/nvidia-container-toolkit.list
      args:
        creates: /etc/apt/sources.list.d/nvidia-container-toolkit.list

    - name: Update apt cache
      apt:
        update_cache: yes

    - name: Install GPU stack + Docker
      apt:
        name:
          - nvidia-open, cuda-toolkit-13-0
          - nvidia-fabricmanager, datacenter-gpu-manager-4-cuda13
          - docker-ce, docker-ce-cli, containerd.io
          - docker-buildx-plugin, docker-compose-plugin
          - nvidia-container-toolkit
        state: present

    - name: Configure Docker for NVIDIA
      command: nvidia-ctk runtime configure --runtime=docker
      notify: restart docker

    - name: Enable services
      systemd:
        name: "{{ item }}"
        enabled: yes
        state: started
      loop:
        - nvidia-fabricmanager
        - nvidia-dcgm
        - docker

    - name: Add CUDA to PATH & LD_LIBRARY_PATH
      blockinfile:
        path: /etc/profile.d/cuda.sh
        create: yes
        marker: "# {mark} CUDA ENV"
        block: |
          export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}
          export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    - name: Load ib_umad module
      lineinfile:
        path: /etc/modules
        line: ib_umad
        create: yes

- name: Master-only tasks (copy tools, shared dirs)
  hosts: master
  become: yes
  tasks:
    - name: Create /opt/testing/tools
      file:
        path: /opt/testing/tools
        state: directory
        mode: '0755'

    - name: Copy your tools (adjust path on your laptop!)
      copy:
        src: /path/to/your/local/tools/   # CHANGE THIS
        dest: /opt/testing/tools/
        mode: preserve
      delegate_to: localhost

    - name: Distribute tools to all clients
      synchronize:
        src: /opt/testing/tools/
        dest: /opt/testing/tools/
      delegate_to: "{{ item }}"
      loop: [client01, client02, client03]

    - name: Create shared NFS export (optional)
      lineinfile:
        path: /etc/exports
        line: "/opt/testing *(rw,sync,no_root_squash)"
        create: yes
      notify: restart nfs

  handlers:
    - name: restart docker
      systemd: name=docker state=restarted
    - name: restart nfs
      systemd: name=nfs-kernel-server state=restarted
```

#### 4. `quick_check.sh` – Run after deployment
```bash
#!/bin/bash
echo "Quick health check on all nodes..."

ansible all -i inventory.yml -m shell -a "hostname && nvidia-smi -L | wc -l && ibstat | grep -i state"
ansible all -i inventory.yml -m shell -a "dcgmi discovery -l"
ansible gpu_nodes -i inventory.yml -m shell -a "nvbandwidth -t" --one-line
ansible gpu_nodes -i inventory.yml -m shell -a "docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi"
```

### How to Use (3 Steps)

1. **On your laptop/bastion**:
   ```bash
   mkdir hpc-cluster && cd hpc-cluster
   wget -O deploy_cluster.sh https://raw.githubusercontent.com/yourname/hpc/main/deploy_cluster.sh
   wget -O inventory.yml https://...
   wget -O full_deploy.yml https://...
   wget -O quick_check.sh https://...
   chmod +x *.sh
   ```

2. **Edit one line** in `full_deploy.yml`:
   ```yaml
   src: /home/youruser/tools/   # ← CHANGE TO YOUR LOCAL TOOLS PATH
   ```

3. **Deploy**:
   ```bash
   ./deploy_cluster.sh
   # Wait ~15–20 minutes
   ./quick_check.sh
   ```

Done. Your 4-node cluster is fully deployed, GPU-ready, Docker-ready, tools copied, and verified.

Run any test:
```bash
ssh master
cd /opt/testing/tools/nccl-tests && make && ./build/all_reduce_perf -b 8 -e 128M -f 2
```





# Dry-run first (very recommended)
ansible-playbook -i inventory.yml deploy-nvidia.yml --check

# Real run
ansible-playbook -i inventory.yml deploy-nvidia.yml -K   # -K asks for sudo password if needed

Optional enhancements

Add NVLink check after install:YAML- name: Verify NVLink status
  command: nvidia-smi nvlink -s
  register: nvlink_status
  changed_when: false

- debug:
    msg: "{{ nvlink_status.stdout_lines }}"
Reboot after driver install (if needed):YAML- name: Reboot if driver was installed
  reboot:
    msg: "Rebooting after NVIDIA driver installation"
  when: "'nvidia-open' in ansible_facts.packages"