# Train all P3 models
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "Training P3 Models (4 algorithms)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow

# BC
Write-Host "`n[1/4] Training P3-BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo bc --process p3 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P3-BC completed" -ForegroundColor Green } else { Write-Host "[FAILED] P3-BC failed" -ForegroundColor Red }

# TD3+BC
Write-Host "`n[2/4] Training P3-TD3BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo td3bc --process p3 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P3-TD3BC completed" -ForegroundColor Green } else { Write-Host "[FAILED] P3-TD3BC failed" -ForegroundColor Red }

# CQL
Write-Host "`n[3/4] Training P3-CQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo cql --process p3 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P3-CQL completed" -ForegroundColor Green } else { Write-Host "[FAILED] P3-CQL failed" -ForegroundColor Red }

# IQL
Write-Host "`n[4/4] Training P3-IQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo iql --process p3 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P3-IQL completed" -ForegroundColor Green } else { Write-Host "[FAILED] P3-IQL failed" -ForegroundColor Red }

Write-Host "`n======================================" -ForegroundColor Yellow
Write-Host "P3 Training Complete" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow
