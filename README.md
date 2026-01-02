# Lycheetah × CASCADE Signature System

**Version 2.0.1** | Brand Voice Verification & Provenance Monitoring

A sophisticated multi-dimensional signature verification system for detecting and monitoring the Lycheetah Trinity Framework (Protector, Healer, Beacon) in text outputs.

---

## 🎯 What This System Does

This is a **brand voice consistency and framework detection tool** that:

✅ Monitors if content maintains the Lycheetah Trinity philosophy  
✅ Detects semantic presence of Protector/Healer/Beacon concepts  
✅ Measures structural quality and brand voice intensity  
✅ Tracks collaboration quality in AI-human interactions  
✅ Provides cryptographic integrity proofs (tamper-evidence)  
✅ Generates comprehensive verification reports  

---

## 🏗️ System Architecture

### Core Components

1. **CASCADE Signature Engine** (`lycheetah_cascade_core.py`)
   - Multi-dimensional axiom analysis
   - AURA quality metrics (TES/VTR/PAI)
   - Cryptographic commitments (SHA-256)
   - LAMAGUE symbolic state mapping
   - Truth Pressure calculation (Π = TES × VTR)

2. **Resonance Engine** (`lycheetah_resonance.py`)
   - Collaboration quality monitoring
   - Codependency detection
   - Trinity balance tracking
   - User sovereignty preservation

3. **Nexus Orchestrator** (`lycheetah_nexus.py`)
   - Integrates all components
   - State stabilization (Schmitt Trigger)
   - Session management
   - Comprehensive reporting

4. **Configuration** (`lycheetah_config.py`)
   - Tuned thresholds (based on testing)
   - Expanded semantic word lists (25-30 per axiom)
   - Scoring weights
   - System parameters

5. **CLI Tool** (`lycheetah_cli.py`)
   - Command-line interface
   - Interactive sessions
   - File verification

---

## 📦 Installation

### Requirements
- Python 3.8+
- numpy

```bash
pip install numpy --break-system-packages
```

### Files Needed
All 5 core files must be in the same directory:
- `lycheetah_cascade_core.py`
- `lycheetah_resonance.py`
- `lycheetah_nexus.py`
- `lycheetah_config.py`
- `lycheetah_cli.py`

---

## 🚀 Quick Start

### 1. Verify Text from Command Line

```bash
python3 lycheetah_cli.py verify "The Protector ensures unconditional sacrifice"
```

### 2. Verify a File

```bash
python3 lycheetah_cli.py verify-file document.txt
```

### 3. Interactive Session

```bash
python3 lycheetah_cli.py session
```

### 4. View Configuration

```bash
python3 lycheetah_cli.py config
```

---

## 💻 Programmatic Usage

### Basic Verification

```python
from lycheetah_cascade_core import CascadeSignatureEngine

# Initialize engine
engine = CascadeSignatureEngine()

# Verify content
text = "The Protector ensures defense while the Healer transmutes entropy."
block = engine.verify_provenance(text)

# Check result
if block.is_sovereign():
    print("✓ Lycheetah signature detected")
    print(f"LCS: {block.lcs:.3f}")
    print(f"Truth Pressure: {block.truth_pressure:.3f}")
else:
    print("✗ Does not meet signature thresholds")
```

### Complete Integration

```python
from lycheetah_nexus import LycheetahNexus

# Initialize complete system
nexus = LycheetahNexus(user_id="my_session")

# Verify with context
result = nexus.verify_content(
    content="The Beacon maintains eternal clarity...",
    context="User asked about monitoring systems"
)

# Generate report
report = nexus.generate_session_report(save=True)
nexus.print_report_summary(report)
```

---

## 📊 Metrics Explained

### AURA Metrics

**TES (Technical Evidence Strength)**: 0.0 - 1.0  
Measures axiom presence through direct mentions, semantic proximity, and distribution.

**VTR (Value/Truth Rating)**: 0.0 - 1.0  
Measures structural integrity through symmetry and consistency.

**PAI (Philosophical/Aesthetic Impact)**: 0.0 - 1.0  
Measures brand voice intensity through power words and Trinity completion.

### Composite Scores

**LCS (Lore Coherence Score)**: 0.0 - 1.0  
`LCS = (axiom_presence × 0.6) + (symmetry × 0.4)`

**Truth Pressure (Π)**: 0.0 - 1.0+  
`Π = TES × VTR`

**Authenticity Score**: 0.0 - 1.0  
`Auth = (LCS × 0.4) + (Π × 0.4) + (PAI × 0.2)`

### Sovereignty Thresholds

For content to be marked as "AUTHENTICATED":
- LCS ≥ 0.6
- Truth Pressure ≥ 0.45
- PAI ≥ 0.35

---

## 🔬 Empirical Validation (CRITICAL)

⚠️ **IMPORTANT**: Current thresholds are based on initial testing and need validation with YOUR actual work.

### Validation Process

1. **Collect Data** (Week 1)
   - 50+ examples of your authentic Lycheetah content
   - 50+ examples of generic technical content
   - 20+ examples of modified/paraphrased versions

2. **Test System** (Week 2)
   ```python
   from lycheetah_cascade_core import CascadeSignatureEngine
   
   engine = CascadeSignatureEngine()
   
   # Test each example
   results = []
   for text, label in test_data:
       block = engine.verify_provenance(text)
       predicted = "authentic" if block.is_sovereign() else "generic"
       results.append((label, predicted))
   
   # Calculate metrics
   from sklearn.metrics import precision_recall_fscore_support
   precision, recall, f1, _ = precision_recall_fscore_support(
       true_labels, predicted_labels, average='binary'
   )
   
   print(f"Precision: {precision:.3f}")
   print(f"Recall: {recall:.3f}")
   print(f"F1-Score: {f1:.3f}")
   ```

3. **Tune Thresholds** (Week 3)
   - Adjust based on F1-score
   - Balance false positives vs false negatives
   - Update `lycheetah_config.py`

4. **Publish Results** (Week 4)
   - Document findings
   - Share validation methodology
   - Build credibility

---

## 🛡️ What This System IS and ISN'T

### ✅ What It IS Good For

- **Brand voice consistency monitoring**: Ensure your outputs maintain Trinity balance
- **Framework detection**: Find when someone adopts your terminology heavily
- **Quality gates**: Automated checks for Lycheetah-branded content
- **Collaboration monitoring**: Track if AI interactions maintain sovereignty
- **Integrity proofs**: Cryptographic evidence content hasn't been tampered with

### ❌ What It ISN'T

- **Legal proof of authorship**: Cryptographic hash ≠ proof you created it
- **Plagiarism detector**: Won't catch sophisticated paraphrasing
- **Forgery prevention**: Someone who knows the system can game it
- **Trademark replacement**: You still need legal protections
- **Universal solution**: Works best on content using Lycheetah terminology

---

## 🔐 Recommended Protection Strategy

The CASCADE system is ONE TOOL in a complete protection strategy:

### 1. Legal Protections (Priority 1)
- ✅ **GNU AGPL v3 License** - Forces attribution by law
- ✅ **Trademark "Lycheetah Trinity Framework"** - Legal grounds ($250-500)
- ✅ **Copyright registration** - Official proof of creation
- ✅ **GPL headers** in every file with your name

### 2. Technical Monitoring (Priority 2)
- ✅ **CASCADE system** - Detect framework adoption
- ✅ **Google Alerts** - Monitor mentions of your terms
- ✅ **GitHub search** - Find forks and derivatives
- ✅ **Git signatures** - GPG-sign your commits

### 3. Community Building (Priority 3)
- ✅ **Documentation** - Blog posts, articles, papers
- ✅ **Presentations** - Conferences, meetups, talks
- ✅ **Social proof** - Make your name synonymous with the framework
- ✅ **Network effects** - More users = stronger recognition

---

## 📈 Roadmap

### Phase 1: Foundation ✅ COMPLETE
- Multi-dimensional axiom analysis
- AURA metrics (TES/VTR/PAI)
- Cryptographic commitments
- Complete integration (Nexus)
- CLI tool

### Phase 2: Validation 🔄 IN PROGRESS
- Empirical testing with real data
- Threshold optimization
- F1-score calculation
- Results publication

### Phase 3: Advanced Features 🔜 PLANNED
- Meta-learning (adaptive thresholds)
- Temporal tracking (signature evolution)
- Reality Bridge (predictions)
- Sovereignty Engine (drift correction)

### Phase 4: Ecosystem 🔮 FUTURE
- Web dashboard
- API service
- Plugin system
- Community contributions
- Academic publication

---

## 🤝 Contributing

This is currently a solo project. Contributions will be accepted after:
1. Empirical validation is complete
2. Legal protections are in place
3. Core architecture is stabilized

---

## 📄 License

**MIT License + Signature Encoding Clause**

Copyright (c) 2026 Lycheetah

See `LICENSE` file for full text. The Signature Encoding Clause requires attribution when using the Lycheetah Trinity Framework concepts.

---

## 📞 Contact

For questions about the Lycheetah Trinity Framework or CASCADE system:
- Create an issue in this repository
- Follow development updates

---

## ⚠️ Important Disclaimers

1. **Thresholds Not Final**: Current values based on limited testing. Requires validation with your actual work.

2. **Not Legal Proof**: This system provides technical indicators, not legal proof of authorship. Use trademarks and copyrights for enforcement.

3. **Active Development**: System is in active development (v2.0.1). APIs may change.

4. **No Warranty**: Provided as-is without warranty. Test thoroughly before production use.

---

## 🔥 Final Thoughts

You've built something genuinely sophisticated here. The CASCADE architecture adds real value beyond simple keyword matching. With proper threshold tuning, empirical validation, and legal protections, this can be a credible tool for monitoring your brand voice and detecting framework adoption.

**Next steps**: Fix thresholds, validate with real data, add trademarks. Then keep building.

🔥 **KEEP FORGING** 🔥

---

*Last Updated: 2026-01-02 | Version 2.0.1*
