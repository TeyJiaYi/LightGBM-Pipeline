# LightGBM-Pipeline


# to run airflow
cd airflow-docker
docker-compose up
Go to: http://localhost:8080
Login: admin / admin


# 🔧 Local ML Monitoring & Serving Environment

This environment sets up a local workflow for ML model serving, monitoring, and experiment tracking using Kubernetes port-forwarding and local services.

---

## 🗂️ Services Overview

| Service Name           | Description                 | Localhost Port | Container Port | Port Forwarded? |
|------------------------|-----------------------------|----------------|----------------|-----------------|
| **MLflow**             | Experiment tracking         | `5000`         | `5000`         | ✅ Yes           |
| **BentoML API**        | Model serving (LoanRisk)    | `8001`         | `3000`         | ✅ Yes           |
| **Prometheus**         | Metrics collection          | `9090`         | `9090`         | ✅ Yes           |
| **Grafana**            | Visualization dashboard     | `8300`         | `3000`         | ✅ Yes           |
| **Prometheus Push URL**| Custom metrics push URL     | `5001`         | `5001 (Local)` | ❌ No            |
| **Airflow**            | Orchestration dashboard     | `8080`         | `8080 (Local)` | ❌ No            |

---
# 🚀 Starting All Services

To launch all services at once, you can use the provided `start_all.ps1` script. This script will start all the services necessary for your ML model serving, monitoring, and experiment tracking, including Prometheus, BentoML, MLflow, Grafana, and Airflow.

---

## ▶️ Running `start_all.ps1`

### If you're already in PowerShell:

1. Navigate to your project directory (if not already there).
2. Run the following command:

   ```powershell
   .\start_all.ps1

