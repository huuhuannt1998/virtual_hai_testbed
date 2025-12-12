# Train P1 Detection Model
Write-Host "Training P1 Detection Model..." -ForegroundColor Cyan
python -m hai_ml.detection.train_detector --process p1 --version 21.03 --epochs 50
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P1 Detection completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P1 Detection failed" -ForegroundColor Red
}
