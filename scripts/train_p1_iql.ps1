# Train P1 - IQL
Write-Host "Training P1-IQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo iql --process p1 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P1-IQL completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P1-IQL failed" -ForegroundColor Red
}
