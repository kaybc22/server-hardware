
server-hardware--engineering-profile/
├── README.md
├── docs/
│   |── architecture_diagrams/
|   |── components_Sepcs/
|   |── validation_process/
├── scripts/
│   ├── stress_test/
│   │   ├── __init__.py
│   │   ├── cpu_memory_test.py
│   │   ├── nvme_test.py
│   │   ├── gpu_test.py
│   │   └── network_test.py
│   ├── monitoring/
│   │   ├── lspci
│   │   ├── node_exporter_setup
│   │   ├── grafana_dashboard
│   │   └── prometheus_config
│   ├── provisioning/
│   │   ├── os_install
│   │   ├── storage_mountpoint
│   │   ├── driver_linux_utls
│   │   └── firmware_managment
│   └── validation/
│       ├── hardware_info.py
│       ├── pcie_validation.py
│       ├── gpu_validation.py
│       └── network_validation.py
├── configs/
│   ├── prometheus.yml
│   ├── grafana.json
│   └── pxe_config.cfg
├── tools/
│   ├── linux_and_open_source
│   ├── proprietary_tools
│   └── internal_tools
└── Performances/
    ├── open_soruce/
    ├── NGC_container/
    └── Git_Huggingface/
