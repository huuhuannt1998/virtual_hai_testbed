# Train P3 Detection Model
Write-Host "Training P3 Detection Model..." -ForegroundColor Cyan
python -m hai_ml.detection.train_detector --process p3 --version 21.03 --epochs 50
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P3 Detection completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P3 Detection failed" -ForegroundColor Red
}
