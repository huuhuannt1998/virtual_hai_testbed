# Training Scripts

Individual training scripts for safer, controlled model training.

## RL Model Scripts

### Individual Model Scripts

Train one specific model at a time:

```powershell
# P1 models
.\scripts\train_p1_bc.ps1
.\scripts\train_p1_td3bc.ps1
.\scripts\train_p1_cql.ps1
.\scripts\train_p1_iql.ps1
```

### Process-Level Scripts

Train all 4 algorithms for a specific process:

```powershell
.\scripts\train_p2_all.ps1   # All P2 RL models
.\scripts\train_p3_all.ps1   # All P3 RL models
.\scripts\train_p4_all.ps1   # All P4 RL models
```

### Algorithm-Specific Scripts

Train one algorithm across all processes:

```powershell
.\scripts\train_cql_only.ps1   # CQL for P1-P4
```

## Detection Model Scripts

### Individual Detection Models

```powershell
.\scripts\train_detect_p1.ps1   # P1 detector (autoencoder + classifier)
.\scripts\train_detect_p2.ps1   # P2 detector
.\scripts\train_detect_p3.ps1   # P3 detector
.\scripts\train_detect_p4.ps1   # P4 detector
```

### All Detection Models

```powershell
.\scripts\train_detect_all.ps1   # All 4 detectors (~60 min)
```

## Usage

1. **Stop current training**: Press `Ctrl+C` in the terminal
2. **Run individual script**: 
   ```powershell
   .\scripts\train_p1_cql.ps1
   ```
3. **Check results**: Models saved in `models/<process>_<algo>/`

## After Fix

Once the CQL bug is fixed, retry failed models:
```powershell
.\scripts\train_cql_only.ps1
```

## Training Checklist

### RL Models (16 total)

**P1 Models:**
- [X] `.\scripts\train_p1_bc.ps1` → `models/p1_bc/`
- [ ] `.\scripts\train_p1_td3bc.ps1` → `models/p1_td3bc/`
- [X] `.\scripts\train_p1_cql.ps1` → `models/p1_cql/`
- [ ] `.\scripts\train_p1_iql.ps1` → `models/p1_iql/`

**P2 Models:** (or run `.\scripts\train_p2_all.ps1`)
- [ ] P2-BC → `models/p2_bc/`
- [ ] P2-TD3BC → `models/p2_td3bc/`
- [ ] P2-CQL → `models/p2_cql/`
- [ ] P2-IQL → `models/p2_iql/`

**P3 Models:** (or run `.\scripts\train_p3_all.ps1`)
- [ ] P3-BC → `models/p3_bc/`
- [ ] P3-TD3BC → `models/p3_td3bc/`
- [ ] P3-CQL → `models/p3_cql/`
- [ ] P3-IQL → `models/p3_iql/`

**P4 Models:** (or run `.\scripts\train_p4_all.ps1`)
- [ ] P4-BC → `models/p4_bc/`
- [ ] P4-TD3BC → `models/p4_td3bc/`
- [ ] P4-CQL → `models/p4_cql/`
- [ ] P4-IQL → `models/p4_iql/`

### Detection Models (4 total)

**Individual:** (or run `.\scripts\train_detect_all.ps1`)
- [ ] `.\scripts\train_detect_p1.ps1` → `models/p1_detector/`
- [ ] `.\scripts\train_detect_p2.ps1` → `models/p2_detector/`
- [ ] `.\scripts\train_detect_p3.ps1` → `models/p3_detector/`
- [ ] `.\scripts\train_detect_p4.ps1` → `models/p4_detector/`

### Progress Tracking

```powershell
# Check which models exist
Get-ChildItem models -Directory | Select-Object Name

# Count completed models
(Get-ChildItem models -Directory).Count
```

## Estimated Times

### RL Models
- **BC**: ~15 minutes per process
- **TD3+BC**: ~20 minutes per process  
- **CQL**: ~25 minutes per process
- **IQL**: ~20 minutes per process

Total per process (RL): ~80 minutes  
Total all 16 RL models: ~5.3 hours

### Detection Models
- **Detector**: ~15 minutes per process  
Total all 4 detectors: ~60 minutes

### Grand Total
**All 20 models (16 RL + 4 detection): ~6.3 hours**
