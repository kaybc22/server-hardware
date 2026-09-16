package main

import (
    "crypto/tls"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "strings"
    "sync"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "gopkg.in/yaml.v3"
)

// --- Config structures ---
type ModuleConfig struct {
    User       string   `yaml:"user"`
    Pass       string   `yaml:"pass"`
    Host       string   `yaml:"host"`
    Interval   string   `yaml:"interval"`
    SensorURIs []string `yaml:"sensor_uris"`
}

type Config struct {
    Modules map[string]ModuleConfig `yaml:"modules"`
}

// --- Redfish sensor JSON ---
type RedfishSensor struct {
    ReadingType  string  `json:"ReadingType"`
    Reading      float64 `json:"Reading"`
    ReadingUnits string  `json:"ReadingUnits"`
}

// --- Globals ---
var (
    configPath = "custom_ipmi_config.yml"
    metrics    = make(map[string]*prometheus.GaugeVec)
    mu         sync.Mutex
)

// Fetch sensor JSON from Redfish
func fetchRedfishSensor(uri, user, pass string) (*RedfishSensor, error) {
    req, _ := http.NewRequest("GET", uri, nil)
    req.SetBasicAuth(user, pass)

    tr := &http.Transport{
        TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, // skip cert validation
    }
    client := &http.Client{Timeout: 10 * time.Second, Transport: tr}

    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var sensor RedfishSensor
    if err := json.NewDecoder(resp.Body).Decode(&sensor); err != nil {
        return nil, err
    }
    return &sensor, nil
}

// Register/update Prometheus metrics with labels
func updateMetric(moduleName, uri string, sensor *RedfishSensor) {
    mu.Lock()
    defer mu.Unlock()

    metricName := fmt.Sprintf("redfish_%s_reading", strings.ToLower(moduleName))
    gauge, ok := metrics[metricName]
    if !ok {
        gauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{
            Name: metricName,
            Help: "Redfish sensor readings",
        }, []string{"sensor", "type", "units"})
        prometheus.MustRegister(gauge)
        metrics[metricName] = gauge
    }

    gauge.With(prometheus.Labels{
        "sensor": uri,
        "type":   sensor.ReadingType,
        "units":  sensor.ReadingUnits,
    }).Set(sensor.Reading)
}

// Collector loop
func collectSensors(moduleName string, cfg ModuleConfig) {
    interval, err := time.ParseDuration(cfg.Interval)
    if err != nil || interval <= 0 {
        interval = 15 * time.Second
    }

    for {
        for _, uri := range cfg.SensorURIs {
            sensor, err := fetchRedfishSensor(uri, cfg.User, cfg.Pass)
            if err != nil {
                log.Printf("[%s] Error fetching %s: %v", moduleName, uri, err)
                continue
            }
            updateMetric(moduleName, uri, sensor)
            log.Printf("[%s] Collected %s = %f %s", moduleName, sensor.ReadingType, sensor.Reading, sensor.ReadingUnits)
        }
        time.Sleep(interval)
    }
}

func main() {
    // Load config
    cfgFile, err := os.ReadFile(configPath)
    if err != nil {
        log.Fatalf("Failed to read config file: %v", err)
    }

    var config Config
    if err := yaml.Unmarshal(cfgFile, &config); err != nil {
        log.Fatalf("Failed to parse config: %v", err)
    }

    if len(config.Modules) == 0 {
        log.Fatal("No modules found in configuration.")
    }

    // Start collectors
    for name, module := range config.Modules {
        log.Printf("Starting collector for module [%s] (host=%s, interval=%s)", name, module.Host, module.Interval)
        go collectSensors(name, module)
    }

    // Start Prometheus server
    log.Println("Starting Prometheus metrics server on :9290/metrics")
    http.Handle("/metrics", promhttp.Handler())
    http.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
        fmt.Fprintln(w, "Redfish IPMI Exporter is running. Metrics available at /metrics")
    })

    log.Fatal(http.ListenAndServe(":9290", nil))
}
