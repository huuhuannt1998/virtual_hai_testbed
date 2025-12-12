# Train P1 - BC
Write-Host "Training P1-BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo bc --process p1 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P1-BC completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P1-BC failed" -ForegroundColor Red
}
