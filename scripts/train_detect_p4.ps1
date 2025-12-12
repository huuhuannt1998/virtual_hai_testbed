# Train P4 Detection Model
Write-Host "Training P4 Detection Model..." -ForegroundColor Cyan
python -m hai_ml.detection.train_detector --process p4 --version 21.03 --epochs 50
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P4 Detection completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P4 Detection failed" -ForegroundColor Red
}
