# Research Plan: RL-Based Intrusion Detection & Prevention for Industrial Control Systems

## 🎯 Target Venues (Top-Tier)

| Conference | Focus | Deadline |
|------------|-------|----------|
| **IEEE S&P** | Security & Privacy | ~Dec/Feb |
| **USENIX Security** | Systems Security | ~Feb |
| **CCS** | Computer & Communications Security | ~May |
| **NDSS** | Network & Distributed Security | ~May/Jul |
| **ACSAC** | Applied Computing Security | ~Jun |

---

## 📊 Gap Analysis: What's Missing in Current Research?

### Existing Work Limitations

1. **Simulation-only** - Most RL-IDS papers use MATLAB/Simulink, not real PLCs
2. **Detection-only** - Few papers actually *prevent* attacks in real-time
3. **Single attack types** - Limited to one or two attack categories
4. **No hardware validation** - No deployment on actual industrial hardware
5. **Delayed response** - Detection latency not suitable for real-time control

### Our Unique Contributions

| Contribution | Novelty |
|--------------|---------|
| **Real PLC testbed** | Siemens S7-1200 with actual I/O simulation |
| **Detection + Prevention** | Closed-loop RL that blocks attacks |
| **Multi-process attacks** | P1-P4 covering diverse ICS scenarios |
| **Edge deployment** | Raspberry Pi for real-time inference |
| **Comprehensive attack suite** | 6+ attack types (bias, replay, DoS, scaling, ramp, noise) |

---

## 🔬 Research Questions

1. **RQ1**: Can RL-based IDPS detect stealthy attacks in real-time on actual PLC hardware?
2. **RQ2**: What is the trade-off between detection accuracy and response latency?
3. **RQ3**: How does the RL agent generalize to unseen attack patterns?
4. **RQ4**: Can edge deployment (Raspberry Pi) meet real-time requirements (<100ms)?

---

## 🏗️ Proposed System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL-IDPS Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   HAI Plant  │───▶│  Siemens PLC │───▶│  RL Agent    │      │
│  │  (Simulated) │    │  S7-1200     │    │  (DQN/PPO)   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Attack     │    │   State      │    │   Action     │      │
│  │   Injection  │    │   Extraction │    │   Execution  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Methodology Phases

### Phase 1: Baseline & Dataset (Weeks 1-3)

| Task | Deliverable |
|------|-------------|
| Generate normal operation data | 24-48 hours of clean data |
| Generate attack scenarios | All 6 attack types × 4 processes |
| Statistical analysis | Mean, std, correlations per tag |
| Create train/val/test splits | 70/15/15 split |

### Phase 2: RL Algorithm Development (Weeks 4-7)

| Task | Options to Evaluate |
|------|---------------------|
| State representation | Raw values vs. statistical features vs. temporal (LSTM) |
| Action space design | Discrete (5 actions) vs. continuous blocking |
| Reward function | Detection reward, false positive penalty, latency penalty |
| RL algorithms | **DQN**, **PPO**, **SAC**, **A2C** (compare all) |

### Phase 3: Training & Optimization (Weeks 8-10)

| Task | Metrics |
|------|---------|
| Hyperparameter tuning | Grid search / Bayesian optimization |
| Ablation studies | Which features matter most? |
| Transfer learning | Train on P1, test on P2-P4 |
| Adversarial robustness | Can attacker evade detection? |

### Phase 4: Real-World Validation (Weeks 11-13)

| Task | Hardware |
|------|----------|
| PLC deployment | Siemens S7-1200 |
| Edge deployment | Raspberry Pi 4 |
| Latency measurement | End-to-end response time |
| Live attack scenarios | Real-time detection demo |

### Phase 5: Paper Writing (Weeks 14-16)

| Section | Key Points |
|---------|------------|
| Introduction | ICS security gap, RL opportunity |
| Related Work | Compare with existing IDS approaches |
| System Design | HAI testbed + RL architecture |
| Evaluation | Metrics, baselines, results |
| Discussion | Limitations, future work |

---

## 📈 Evaluation Metrics

### Detection Performance

| Metric | Target | Description |
|--------|--------|-------------|
| **Detection Rate (DR)** | >95% | True positive rate |
| **False Positive Rate (FPR)** | <5% | False alarms |
| **F1-Score** | >0.90 | Harmonic mean of precision/recall |
| **AUC-ROC** | >0.95 | Area under ROC curve |

### System Performance

| Metric | Target | Description |
|--------|--------|-------------|
| **Detection Latency** | <100ms | Time from attack to detection |
| **Response Latency** | <200ms | Time from attack to mitigation |
| **Throughput** | >100 decisions/sec | Inference speed |
| **Memory Usage** | <512MB | For edge deployment |

### RL-Specific

| Metric | Description |
|--------|-------------|
| **Cumulative Reward** | Total reward per episode |
| **Convergence Time** | Episodes to stable policy |
| **Generalization** | Performance on unseen attacks |

---

## 🔄 Baseline Comparisons

Compare RL-IDPS against:

1. **Rule-based IDS** - Threshold-based detection
2. **Statistical methods** - Z-score, CUSUM, EMA
3. **Traditional ML** - Random Forest, SVM, XGBoost
4. **Deep Learning** - LSTM autoencoder, CNN
5. **Existing RL-IDS** - From literature (cite and compare)

---

## 🧪 Experiment Design

### Experiment 1: Detection Accuracy

- **Setup**: 1000 attack samples per type, 1000 normal samples
- **Compare**: All RL algorithms vs. baselines
- **Metric**: DR, FPR, F1, AUC-ROC

### Experiment 2: Response Time

- **Setup**: Real-time attack injection
- **Measure**: Detection latency, action latency
- **Hardware**: PC vs. Raspberry Pi

### Experiment 3: Generalization

- **Setup**: Train on P1+P2, test on P3+P4
- **Measure**: Cross-process detection accuracy
- **Variation**: Train on 3 attacks, test on remaining 3

### Experiment 4: Adversarial Robustness

- **Setup**: Attacker knows detection model
- **Attack**: Evasion attacks, gradual drift
- **Measure**: Detection under adversarial conditions

---

## 📝 Expected Results Table (for paper)

| Method | DR(%) | FPR(%) | F1 | Latency(ms) |
|--------|-------|--------|-----|-------------|
| Threshold | ~70 | ~15 | ~0.75 | <10 |
| Random Forest | ~85 | ~8 | ~0.85 | <50 |
| LSTM-AE | ~88 | ~6 | ~0.88 | <100 |
| **DQN (Ours)** | ~95 | ~4 | ~0.93 | <80 |
| **PPO (Ours)** | ~96 | ~3 | ~0.94 | <80 |

---

## ❓ Key Decisions Needed

Before coding, we should decide:

### 1. Which RL algorithm to focus on?

- **DQN** - Simple, well-understood
- **PPO** - State-of-the-art, stable
- **SAC** - Sample efficient
- **Multi-agent RL** - One per process?

### 2. State representation?

- Raw sensor values only
- Statistical features (mean, std, rate of change)
- Temporal features (LSTM/Transformer)
- Graph-based (process relationships)

### 3. Action space?

- **Simple**: Alert only
- **Medium**: Alert + Block
- **Full**: Alert + Soft Block + Hard Block + Restore

### 4. Reward function design?

- **Detection-focused**: +1 for true positive
- **Prevention-focused**: +10 for blocking attack
- **Safety-focused**: -100 for false positive that blocks normal operation

### 5. Training approach?

- **Offline** - Pre-collected dataset
- **Online** - Learn during operation
- **Hybrid** - Offline pre-training + online fine-tuning

---

## 🚀 Implementation Roadmap

Once decisions are made:

1. ☐ Refactor the RL module based on chosen approach
2. ☐ Create proper experiment scripts
3. ☐ Add comprehensive logging/metrics
4. ☐ Build comparison baselines
5. ☐ Set up proper train/val/test data generation
6. ☐ Run experiments and collect results
7. ☐ Write paper

---

## 📚 Related Work to Review

### RL for IDS/IPS

- [ ] Deep Q-Learning for Intrusion Detection
- [ ] PPO-based Network Intrusion Detection
- [ ] Multi-Agent RL for Distributed IDS

### ICS Security

- [ ] HAI Dataset papers (original)
- [ ] SCADA security surveys
- [ ] PLC attack taxonomies

### Real-time Edge Deployment

- [ ] TinyML for anomaly detection
- [ ] Edge-based inference optimization
- [ ] Raspberry Pi ML benchmarks

---

## 📧 Notes

- **Conference submission requires**: Novel contribution, strong evaluation, reproducibility
- **Code release**: Plan to open-source the testbed + RL framework
- **Demo video**: Show real-time attack detection on actual PLC

---

*Last updated: December 2024*
