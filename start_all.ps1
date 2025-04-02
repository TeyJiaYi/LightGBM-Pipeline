Write-Host "▶️ Starting Prometheus Push endpoint..."
Start-Process powershell -ArgumentList "Start-Process 'python' -ArgumentList 'push_gateway.py'"

Start-Sleep -Seconds 1

Write-Host "▶️ Starting Airflow UI..."
Start-Process powershell -ArgumentList "Start-Process 'airflow webserver --port 8080'"

Start-Sleep -Seconds 2

Write-Host "▶️ Starting MLflow tracking server on localhost:5000..."
Start-Process powershell -ArgumentList "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"

Start-Sleep -Seconds 2

Write-Host "⏩ Port-forwarding BentoML API → localhost:8001"
Start-Process powershell -ArgumentList "kubectl port-forward svc/loanrisk-service -n ml 8001:3000"

Write-Host "⏩ Port-forwarding Prometheus → localhost:9090"
Start-Process powershell -ArgumentList "kubectl port-forward svc/prometheus -n monitoring 9090:9090"

Write-Host "⏩ Port-forwarding Grafana → localhost:8300"
Start-Process powershell -ArgumentList "kubectl port-forward svc/grafana -n monitoring 8300:3000"

Write-Host "✅ All services launched. Check browser tabs."
