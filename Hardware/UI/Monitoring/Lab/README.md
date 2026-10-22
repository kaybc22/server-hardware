#SOP
#Download the project
git clone https://github.com/kaybc22/server-hardware.git
#run the scipt to install all HGX package
Node_setup.sh

#monitor
#Install all the services in the system that want to monitor
#custom node_exported --> prometheus --> grafana UI
#Node_exporter
node_exp_version=v1.12.1
wget https://github.com/prometheus/node_exporter/releases/download/v1.12.1/node_exporter-1.12.1.linux-amd64.tar.gz
tar -xzvf node_exporter-1.12.1.linux-amd64.tar.gz
cd node_exporter-1.12.1.linux-amd64; ./node_exporter  #http://os_ip:9100/metrics
 
#Prometheus
Prometheus_version=v3.15.1
wget https://github.com/prometheus/prometheus/releases/download/v3.15.1/prometheus-3.15.1.linux-amd64.tar.gz
tar -xzvf prometheus-3.15.1.linux-amd64.tar.gz; cd prometheus-3.15.1.linux-amd64
./prometheus --config.file=./prometheus.yml

#config
global:
  scrape_interval: 15s  # By default, scrape targets every 15 seconds.
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    static_configs:
      - targets: ['172.31.36.80:9090'] #IP of the target system 

  - job_name: 'node_exporter'
    static_configs:
     - targets: ['172.31.36.80:9100'] #http://os_ip:9090

#Grafana
sudo apt-get install -y musl apt-transport-https 
#.deb package
wget https://dl.grafana.com/grafana/release/13.2.1/grafana_13.2.1_33191028959_linux_amd64.deb
sudo dpkg -i grafana_13.2.1_33191028959_linux_amd64.deb
#repo
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
sudo chmod 644 /etc/apt/keyrings/grafana.gpg
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list

#sudo ufw allow 3000/tcp  For Ubuntu/Debian systems
sudo apt-get update; sudo apt-get install grafana -y ; sudo systemctl start grafana-server; sudo systemctl enable grafana-server 
#http://os_ip:3000 #admin/admin --> Add new connection (prometheus) --> http://os_ip:9090 (customize the connection) 

#Grafana 
Build new dashboard or import the (Dashboards --> Import dashboard Json)

