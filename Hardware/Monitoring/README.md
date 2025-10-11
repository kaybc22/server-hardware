##Install docker 
# Add Docker's official GPG key:
sudo apt-get update; sudo apt-get install ca-certificates curl; sudo install -m 0755 -d /etc/apt/keyrings; sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc; sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null  ; sudo apt-get update
# To install the latest version
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
# Verify that the installation
sudo docker run hello-world
###note: sudo apt-get remove pigz

#To run Docker without root privileges
sudo groupadd docker; sudo usermod -aG docker $USER; docker run hello-world

##toolkit Install and run the sample GPU with docker (nvidia-container-toolkit)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list; sudo apt-get update; sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker; sudo systemctl restart docker
#runtime
systemctl list-units | grep -E 'docker|containerd|crio'
ls /usr/bin | grep -E 'runc|containerd|crio|dockerd'


#config file /etc/docker/daemon.json
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
docker run --gpus all --rm -it -v /container/volumes/dcgm:/usr/local/dcgm:rw gpu-burn 60

docker run -d --gpus all --cap-add SYS_ADMIN --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.1-ubuntu22.04