# Train P1 - TD3+BC
Write-Host "Training P1-TD3BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo td3bc --process p1 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P1-TD3BC completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P1-TD3BC failed" -ForegroundColor Red
}
