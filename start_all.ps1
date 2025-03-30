Write-Host "▶️ Starting MLflow tracking server..."
Start-Process powershell -ArgumentList "mlflow server --host 127.0.0.1 --port 5000"

Start-Sleep -Seconds 2

Write-Host "⏩ Port-forwarding MLflow → localhost:8500"
Start-Process powershell -ArgumentList "kubectl port-forward svc/mlflow-service -n ml 8500:5000"

Write-Host "⏩ Port-forwarding BentoML API → localhost:8001"
Start-Process powershell -ArgumentList "kubectl port-forward svc/loanrisk-service -n ml 8001:3000"

Write-Host "⏩ Port-forwarding Prometheus → localhost:9091"
Start-Process powershell -ArgumentList "kubectl port-forward svc/prometheus -n monitoring 9091:9090"

Write-Host "⏩ Port-forwarding Grafana → localhost:8300"
Start-Process powershell -ArgumentList "kubectl port-forward svc/grafana -n monitoring 8300:3000"

Write-Host "✅ All services launched. Check browser tabs."
