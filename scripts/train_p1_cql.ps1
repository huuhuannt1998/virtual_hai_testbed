# Train P1 - CQL
Write-Host "Training P1-CQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo cql --process p1 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P1-CQL completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P1-CQL failed" -ForegroundColor Red
}
