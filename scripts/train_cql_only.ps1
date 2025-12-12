# Train only CQL models (for retry after fix)
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "Training CQL Models Only (4 processes)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow

$processes = @("p1", "p2", "p3", "p4")
$i = 1

foreach ($proc in $processes) {
    Write-Host "`n[$i/4] Training $proc-CQL..." -ForegroundColor Cyan
    python -m hai_ml.rl.train_real --algo cql --process $proc --version 21.03 --epochs 100
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] $proc-CQL completed" -ForegroundColor Green
    } else {
        Write-Host "[FAILED] $proc-CQL failed" -ForegroundColor Red
    }
    $i++
}

Write-Host "`n======================================" -ForegroundColor Yellow
Write-Host "CQL Training Complete" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow
