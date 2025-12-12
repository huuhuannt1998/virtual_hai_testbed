# Train All Detection Models
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "Training Detection Models (4 processes)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow

$processes = @("p1", "p2", "p3", "p4")
$i = 1

foreach ($proc in $processes) {
    Write-Host "`n[$i/4] Training $proc Detection..." -ForegroundColor Cyan
    python -m hai_ml.detection.train_detector --process $proc --version 21.03 --epochs 50
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] $proc Detection completed" -ForegroundColor Green
    } else {
        Write-Host "[FAILED] $proc Detection failed" -ForegroundColor Red
    }
    $i++
}

Write-Host "`n======================================" -ForegroundColor Yellow
Write-Host "Detection Training Complete" -ForegroundColor Yellow
Write-Host "Total time: ~60 minutes (4 × 15 min)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow
