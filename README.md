# DeepSeekMath-V2 Implementation

[![Tests](https://img.shields.io/badge/tests-131%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

A clean, modular implementation of [DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning](https://github.com/deepseek-ai/DeepSeek-Math-V2) for learning distributed training, reinforcement learning (GRPO), and mathematical reasoning systems.

---

## 🎯 Project Goals

This implementation focuses on **learning and understanding** the architecture and training methodology of DeepSeekMath-V2:

1. **Understand the three-model system** (Verifier, Meta-Verifier, Generator)
2. **Learn distributed training** (DeepSpeed, Megatron-LM patterns)
3. **Implement GRPO** (Group Relative Policy Optimization)
4. **Explore self-verification** mechanisms in LLMs
5. **Practice ML engineering** best practices (modular code, testing, documentation)

**Note:** This is a **learning implementation** with a limited GPU budget (1x T4, 16GB). For production-scale training, see the official DeepSeek repository.

---

## 📊 What is DeepSeekMath-V2?

DeepSeekMath-V2 is a system for training LLMs to solve mathematical proofs with **self-verification**:

### **The Three-Model System:**

```
┌─────────────┐
│  Verifier   │ ← Scores proofs (0, 0.5, 1)
│    (π_φ)    │   Identifies errors
└─────────────┘
       ↓
┌─────────────┐
│Meta-Verifier│ ← Checks verifier quality
│    (π_η)    │   Prevents hallucinations
└─────────────┘
       ↓
┌─────────────┐
│  Generator  │ ← Generates proofs
│    (π_θ)    │   Self-verifies & refines
└─────────────┘
```

### **Key Innovation:**
Models learn to **verify their own reasoning** and iteratively improve proofs until no issues remain.

### **Results (from paper):**
- 🥇 Gold medal: IMO 2025, CMO 2024
- 📈 118/120 on Putnam 2024 (vs. human max of 90)

---

## 🚀 Quick Start

### **1. Setup Environment**

```bash
# Clone repository
git clone https://github.com/your-username/deepseek-math-v2
cd deepseek-math-v2

# Run automated setup
chmod +x scripts/00_setup_environment.sh
bash scripts/00_setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### **2. Run Tests**

```bash
# Run all tests (131 tests)
pytest tests/ -v

# Expected output: 131 passed
```

### **3. Try the Modules**

```python
# Example: Generate a prompt
from src.utils.prompts import get_proof_generation_prompt

problem = "Prove that sqrt(2) is irrational"
prompt = get_proof_generation_prompt(problem)
print(prompt)

# Example: Load configuration
from src.utils.config_loader import create_default_config

config = create_default_config("my_experiment")
print(f"Learning rate: {config.training.learning_rate}")

# Example: Compute rewards
from src.training.reward_functions import compute_score_reward

reward = compute_score_reward(predicted=0.5, ground_truth=1.0)
print(f"Reward: {reward}")  # 0.5
```

---

## 📁 Repository Structure

```
deepseek-math-v2/
├── README.md                    ← You are here
├── requirements.txt             ← Dependencies
├── setup.py                     ← Package setup
│
├── configs/                     ← YAML configurations
│   ├── model/
│   ├── training/
│   └── data/
│
├── src/                         ← Source code (7 modules, ~1,940 lines)
│   ├── utils/                   ← Utilities
│   │   ├── prompts.py          ← Prompt templates
│   │   ├── config_loader.py    ← Config management
│   │   └── profiling.py        ← GPU profiling (TODO)
│   │
│   ├── data/                    ← Data pipeline
│   │   ├── proof_dataset.py    ← PyTorch Datasets
│   │   └── data_collator.py    ← Batch preparation
│   │
│   ├── model/                   ← Model wrappers
│   │   ├── base_model.py       ← Base model utilities
│   │   ├── model_utils.py      ← Helper functions
│   │   ├── verifier.py         ← Verifier (TODO)
│   │   └── generator.py        ← Generator (TODO)
│   │
│   └── training/                ← Training logic
│       ├── reward_functions.py ← GRPO rewards
│       ├── grpo_trainer.py     ← GRPO implementation (TODO)
│       └── verifier_trainer.py ← Training loops (TODO)
│
├── scripts/                     ← Executable scripts
│   ├── 00_setup_environment.sh  ← Environment setup
│   ├── 01_upgrade_to_gpu.sh     ← GPU upgrade
│   ├── 10_train_verifier.py     ← Training scripts (TODO)
│   ├── 20_auto_label_proofs.py  ← Auto-labeling (TODO)
│   └── 30_evaluate.py           ← Evaluation (TODO)
│
├── tests/                       ← Unit tests (131 passing!)
│   ├── test_prompts.py
│   ├── test_config_loader.py
│   ├── test_reward_functions.py
│   ├── test_proof_dataset.py
│   ├── test_data_collator.py
│   ├── test_base_model.py
│   └── test_model_utils.py
│
├── notebooks/                   ← Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   └── 03_grpo_analysis.ipynb
│
├── experiments/                 ← Experiment outputs
│   ├── logs/
│   ├── checkpoints/
│   └── results/
│
└── docs/                        ← Documentation
    ├── setup_guide.md
    ├── BASE_MODEL_EXPLAINED.md
    ├── DATA_COLLATOR_EXPLAINED.md
    └── MODEL_UTILS_EXPLAINED.md
```

---

## 🧩 Implementation Progress

### ✅ **Completed Modules** (7/12)

| Module | Lines | Tests | Status |
|--------|-------|-------|--------|
| `prompts.py` | 280 | 14 ✓ | ✅ Complete |
| `config_loader.py` | 240 | 9 ✓ | ✅ Complete |
| `reward_functions.py` | 220 | 16 ✓ | ✅ Complete |
| `proof_dataset.py` | 250 | 18 ✓ | ✅ Complete |
| `data_collator.py` | 280 | 20 ✓ | ✅ Complete |
| `base_model.py` | 290 | 20 ✓ | ✅ Complete |
| `model_utils.py` | 380 | 34 ✓ | ✅ Complete |
| **Total** | **1,940** | **131** | **53% done** |

### ⏭️ **Next Steps**

- [ ] `verifier.py` - Proof verification model
- [ ] `generator.py` - Proof generation model
- [ ] `grpo_trainer.py` - GRPO training loop
- [ ] `verifier_trainer.py` - Verifier training
- [ ] `auto_labeling.py` - Automated proof labeling

---

## 🎓 Key Concepts

### **1. Three-Model Training Pipeline**

```
Phase 1: Train Verifier
  ├─ Input: (problem, proof, expert_score)
  ├─ Output: Analysis + Score
  └─ Reward: R_format × R_score

Phase 2: Train Meta-Verifier
  ├─ Input: (problem, proof, verifier_analysis, meta_score)
  ├─ Output: Quality assessment
  └─ Reward: R_format × R_score × R_meta

Phase 3: Enhanced Verifier
  ├─ Use meta-verifier feedback
  └─ Reduce hallucinated issues

Phase 4: Train Generator
  ├─ Input: problem
  ├─ Output: proof + self-analysis
  └─ Reward: α·R_Y + β·R_Z (α=0.76, β=0.24)

Phase 5: Auto-Label & Iterate
  ├─ Generate hard proofs
  ├─ Scale verification (n=64 samples)
  ├─ Auto-label via majority voting
  └─ Retrain verifier → Loop
```

### **2. GRPO (Group Relative Policy Optimization)**

RL algorithm that rewards based on **relative ranking** within a group, not absolute scores. More stable than PPO for mathematical reasoning.

### **3. Self-Verification**

Model generates: `Proof + Self-Analysis + Self-Score`

Incentivized to:
- Identify issues in own work
- Fix issues before finalizing
- Accurately assess proof quality

---

## 🔧 Technical Details

### **Model Architecture**
- **Base:** DeepSeek-V3.2-Exp-Base (MoE, ~236B total params, ~37B active)
- **Context:** 128K tokens
- **Precision:** bfloat16
- **Hardware (paper):** Multi-GPU cluster with DeepSpeed ZeRO-3

### **Our Setup (Learning)**
- **GPU:** 1x NVIDIA T4 (16GB) - **cannot fit full model!**
- **Alternative:** Use DeepSeek-Math-7B or LoRA fine-tuning
- **Budget:** £204 (~$257) for ~500 hours of compute

### **Key Technologies**
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace model loading
- **DeepSpeed** - Distributed training (when GPU available)
- **Pydantic** - Type-safe configuration
- **Pytest** - Testing framework

---

## 📚 Documentation

### **Getting Started**
- [Setup Guide](docs/setup_guide.md) - Environment setup
- [Setup Explanation](docs/setup_explanation.md) - How setup scripts work (interview prep)

### **Module Explanations**
- [BASE_MODEL_EXPLAINED.md](docs/BASE_MODEL_EXPLAINED.md) - Model loading/saving
- [DATA_COLLATOR_EXPLAINED.md](docs/DATA_COLLATOR_EXPLAINED.md) - Batch preparation
- [MODEL_UTILS_EXPLAINED.md](docs/MODEL_UTILS_EXPLAINED.md) - Helper utilities

### **Training**
- [PHASE1_SUMMARY.md](docs/PHASE1_SUMMARY.md) - Initial implementation summary

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_prompts.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run fast tests only
pytest tests/ -m "not slow"
```

**Current Coverage:** 131 tests, all passing ✅

---

## 💡 Design Philosophy

### **1. Modular Architecture**
- Each file ~200-300 lines (not 1000+!)
- Single responsibility per module
- Easy to understand and modify

### **2. Test-Driven Development**
- Tests written alongside code
- 100% of implemented functions tested
- Can develop on CPU, no GPU needed

### **3. Learning-Focused**
- Clear documentation and comments
- Interview prep materials included
- Explanations for "why" not just "what"

### **4. Production Patterns**
- Type hints everywhere
- Pydantic validation
- Proper error handling
- Clean separation of concerns

---

## 🚧 Limitations & Future Work

### **Current Limitations**
- ❌ Full model doesn't fit on single T4 GPU
- ❌ Training not yet implemented (GRPO trainer in progress)
- ❌ No actual proof scraping from AoPS
- ❌ Mock mode for testing (no real model loading)

### **Planned Improvements**
- [ ] LoRA/QLoRA support for T4 training
- [ ] Implement full GRPO training loop
- [ ] Add AoPS data scraper
- [ ] Real-time training monitoring
- [ ] GPU profiling with PyCUDA
- [ ] Multi-GPU support with DeepSpeed

---

## 📖 References

### **Papers**
- [DeepSeekMath-V2 Paper](https://github.com/deepseek-ai/DeepSeek-Math-V2) - Original paper
- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization

### **Related Work**
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) - Base model family
- [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) - Formal theorem proving

---

## 🤝 Contributing

This is a **learning project**, but contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DeepSeek AI** - Original paper and implementation
- **HuggingFace** - Transformers library
- **PyTorch Team** - Deep learning framework

---

## 📊 Project Stats

- **Language:** Python 3.11+
- **Lines of Code:** ~1,940 (source) + ~1,730 (tests)
- **Tests:** 131 passing
- **Modules:** 7 complete, 5 in progress
- **Documentation:** 5 detailed guides
- **License:** MIT