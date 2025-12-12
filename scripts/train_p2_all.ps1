# Train all P2 models
Write-Host "======================================" -ForegroundColor Yellow
Write-Host "Training P2 Models (4 algorithms)" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow

# BC
Write-Host "`n[1/4] Training P2-BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo bc --process p2 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P2-BC completed" -ForegroundColor Green } else { Write-Host "[FAILED] P2-BC failed" -ForegroundColor Red }

# TD3+BC
Write-Host "`n[2/4] Training P2-TD3BC..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo td3bc --process p2 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P2-TD3BC completed" -ForegroundColor Green } else { Write-Host "[FAILED] P2-TD3BC failed" -ForegroundColor Red }

# CQL
Write-Host "`n[3/4] Training P2-CQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo cql --process p2 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P2-CQL completed" -ForegroundColor Green } else { Write-Host "[FAILED] P2-CQL failed" -ForegroundColor Red }

# IQL
Write-Host "`n[4/4] Training P2-IQL..." -ForegroundColor Cyan
python -m hai_ml.rl.train_real --algo iql --process p2 --version 21.03 --epochs 100
if ($LASTEXITCODE -eq 0) { Write-Host "[SUCCESS] P2-IQL completed" -ForegroundColor Green } else { Write-Host "[FAILED] P2-IQL failed" -ForegroundColor Red }

Write-Host "`n======================================" -ForegroundColor Yellow
Write-Host "P2 Training Complete" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Yellow
