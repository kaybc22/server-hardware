
server-hardware--engineering-profile/
├── README.md
├── Docs/
│   │── architecture_diagrams/
│   │── components_Sepcs/
│   └── validation_process/
├── Hardware/
│   ├── Testing/ 
│   │   ├── main.py
│   │   ├── cpu_memory_benchmark.py
│   │   ├── nvme_benchmark.py
│   │   ├── gpu_benchmark.py
│   │   ├── network_benchmark.py
│   │   └── pcie_all.py
│   ├── monitoring/
│   │   ├── lspci
│   │   ├── node_exporter_setup
│   │   ├── grafana_dashboard
│   │   ├── custom_config
│   │   └── prometheus_config
│   ├── provisioning/
│   │   ├── os_install
│   │   ├── storage_mountpoint
│   │   ├── driver_linux_utls
│   │   └── firmware_management
│   └── validation/
│       ├── main.py
│       ├── cpu_validation.py
│       ├── gpu_validation.py
│       ├── storage_validation.py
│       ├── network_validation.py
│       └── validation.py
├── Configs/
│   ├── prometheus.yml
│   ├── grafana.json
│   │── pxe_config.cfg 
│   └── custom.json 
├── Tools/
│   ├── linux_and_open_source
│   ├── proprietary_tools
│   └── internal_tools
└── Performances/
    ├── custom/
    ├── open_soruce/
    ├── NGC_container/
    └── Git_Huggingface/
