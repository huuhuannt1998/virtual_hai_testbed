# Train all P4 models
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "Training P4 Models (4 algorithms)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow

# BC
Write-Host "`n[1/4] Training P4-BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo bc --process p4 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P4-BC completed" -ForegroundColor Green } else { Write-Host "[FAILED] P4-BC failed" -ForegroundColor Red }

# TD3+BC
Write-Host "`n[2/4] Training P4-TD3BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo td3bc --process p4 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P4-TD3BC completed" -ForegroundColor Green } else { Write-Host "[FAILED] P4-TD3BC failed" -ForegroundColor Red }

# CQL
Write-Host "`n[3/4] Training P4-CQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo cql --process p4 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P4-CQL completed" -ForegroundColor Green } else { Write-Host "[FAILED] P4-CQL failed" -ForegroundColor Red }

# IQL
Write-Host "`n[4/4] Training P4-IQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo iql --process p4 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P4-IQL completed" -ForegroundColor Green } else { Write-Host "[FAILED] P4-IQL failed" -ForegroundColor Red }

Write-Host "`n======================================" -ForegroundColor Yellow
Write-Host "P4 Training Complete" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow
