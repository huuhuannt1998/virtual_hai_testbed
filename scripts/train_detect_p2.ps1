# Train P2 Detection Model
Write-Host "Training P2 Detection Model..." -ForegroundColor Cyan
python -m hai_ml.detection.train_detector --process p2 --version 21.03 --epochs 50
if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] P2 Detection completed successfully" -ForegroundColor Green
} else {
    Write-Host "[FAILED] P2 Detection failed" -ForegroundColor Red
}
