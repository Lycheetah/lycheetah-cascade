"""
lycheetah_validation_engine.py
===============================
Empirical Validation Framework for CASCADE System

CRITICAL MODULE: This addresses the #1 issue identified in the analysis:
"No empirical validation data yet"

Provides:
- F1-score, precision, recall calculation
- ROC curves and AUC metrics
- Confusion matrices
- Cross-validation
- Threshold optimization
- Statistical significance testing

Author: Lycheetah × CASCADE
Version: 2.0.0
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        roc_curve,
        roc_auc_score,
        classification_report
    )
    from sklearn.model_selection import KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn not available - install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

from lycheetah_cascade_core import CascadeSignatureEngine
from lycheetah_config import SignatureThresholds


# ============================================================================
# VALIDATION DATA STRUCTURES
# ============================================================================

@dataclass
class ValidationSample:
    """Single validation data point"""
    text: str
    true_label: str  # 'authentic' or 'generic'
    source: Optional[str] = None  # Where this came from
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results from validation run"""
    timestamp: datetime
    dataset_name: str
    sample_count: int
    
    # Classification metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    # Threshold analysis
    optimal_thresholds: Dict[str, float]
    current_thresholds: Dict[str, float]
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # ROC analysis
    auc_score: Optional[float] = None
    roc_curve_data: Optional[Dict] = None
    
    # Detailed predictions
    predictions: List[Dict] = field(default_factory=list)
    
    # Statistical tests
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'dataset_name': self.dataset_name,
            'sample_count': self.sample_count,
            'metrics': {
                'precision': round(self.precision, 4),
                'recall': round(self.recall, 4),
                'f1_score': round(self.f1_score, 4),
                'accuracy': round(self.accuracy, 4)
            },
            'thresholds': {
                'optimal': self.optimal_thresholds,
                'current': self.current_thresholds
            },
            'confusion_matrix': {
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            },
            'auc_score': round(self.auc_score, 4) if self.auc_score else None,
            'confidence_interval': self.confidence_interval,
            'predictions': self.predictions
        }


# ============================================================================
# VALIDATION ENGINE
# ============================================================================

class ValidationEngine:
    """
    Comprehensive validation framework for CASCADE system
    
    Key Features:
    - Empirical testing with ground truth
    - Threshold optimization
    - Cross-validation
    - Statistical significance
    - ROC analysis
    """
    
    def __init__(self, engine: Optional[CascadeSignatureEngine] = None):
        """
        Initialize validation engine
        
        Args:
            engine: Signature engine to validate (creates new if None)
        """
        self.engine = engine if engine else CascadeSignatureEngine()
        self.validation_history: List[ValidationResult] = []
        
    # ------------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------------
    
    def load_dataset(self, 
                     authentic_dir: str,
                     generic_dir: str) -> List[ValidationSample]:
        """
        Load validation dataset from directories
        
        Args:
            authentic_dir: Directory with authentic Lycheetah content
            generic_dir: Directory with generic content
        
        Returns:
            List of validation samples
        """
        samples = []
        
        # Load authentic samples
        auth_path = Path(authentic_dir)
        if auth_path.exists():
            for file in auth_path.glob('*.txt'):
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    samples.append(ValidationSample(
                        text=text,
                        true_label='authentic',
                        source=str(file),
                        metadata={'category': 'authentic'}
                    ))
        
        # Load generic samples
        gen_path = Path(generic_dir)
        if gen_path.exists():
            for file in gen_path.glob('*.txt'):
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    samples.append(ValidationSample(
                        text=text,
                        true_label='generic',
                        source=str(file),
                        metadata={'category': 'generic'}
                    ))
        
        print(f"✓ Loaded {len(samples)} samples")
        print(f"  - Authentic: {sum(1 for s in samples if s.true_label == 'authentic')}")
        print(f"  - Generic: {sum(1 for s in samples if s.true_label == 'generic')}")
        
        return samples
    
    def load_from_json(self, filepath: str) -> List[ValidationSample]:
        """Load validation dataset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append(ValidationSample(
                text=item['text'],
                true_label=item['label'],
                source=item.get('source'),
                metadata=item.get('metadata', {})
            ))
        
        return samples
    
    # ------------------------------------------------------------------------
    # VALIDATION EXECUTION
    # ------------------------------------------------------------------------
    
    def validate(self, 
                samples: List[ValidationSample],
                dataset_name: str = "validation") -> ValidationResult:
        """
        Run complete validation
        
        Args:
            samples: Validation samples
            dataset_name: Name for this validation run
        
        Returns:
            Validation results
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn required for validation - install scikit-learn")
        
        print(f"\n{'='*70}")
        print(f"RUNNING VALIDATION: {dataset_name}")
        print(f"{'='*70}\n")
        
        # Run predictions
        print(f"Testing {len(samples)} samples...")
        predictions = []
        true_labels = []
        scores = []  # For ROC curve
        
        for i, sample in enumerate(samples, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(samples)}")
            
            # Verify with CASCADE
            block = self.engine.verify_provenance(sample.text)
            predicted = 'authentic' if block.is_sovereign() else 'generic'
            
            predictions.append(predicted)
            true_labels.append(sample.true_label)
            scores.append(block.authenticity_score)
        
        # Convert to binary
        y_true = [1 if label == 'authentic' else 0 for label in true_labels]
        y_pred = [1 if pred == 'authentic' else 0 for pred in predictions]
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # ROC curve
        auc = None
        roc_data = None
        try:
            auc = roc_auc_score(y_true, scores)
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            print(f"⚠️ ROC curve calculation failed: {e}")
        
        # Current thresholds
        current_thresholds = SignatureThresholds.get_all()
        
        # Optimize thresholds
        optimal_thresholds = self._optimize_thresholds(samples)
        
        # Create detailed predictions
        detailed_predictions = []
        for sample, pred, score in zip(samples, predictions, scores):
            detailed_predictions.append({
                'text_preview': sample.text[:100] + '...',
                'true_label': sample.true_label,
                'predicted_label': pred,
                'correct': pred == sample.true_label,
                'authenticity_score': round(score, 4),
                'source': sample.source
            })
        
        # Confidence interval (bootstrapping)
        ci = self._compute_confidence_interval(y_true, y_pred)
        
        # Build result
        result = ValidationResult(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            sample_count=len(samples),
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            optimal_thresholds=optimal_thresholds,
            current_thresholds=current_thresholds,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            auc_score=auc,
            roc_curve_data=roc_data,
            predictions=detailed_predictions,
            confidence_interval=ci
        )
        
        # Store in history
        self.validation_history.append(result)
        
        # Print summary
        self._print_validation_summary(result)
        
        return result
    
    # ------------------------------------------------------------------------
    # THRESHOLD OPTIMIZATION
    # ------------------------------------------------------------------------
    
    def _optimize_thresholds(self, 
                            samples: List[ValidationSample]) -> Dict[str, float]:
        """
        Find optimal thresholds by grid search
        
        Returns:
            Dict of optimal threshold values
        """
        print("\nOptimizing thresholds...")
        
        # Test range of thresholds
        lcs_range = np.arange(0.3, 0.8, 0.05)
        truth_range = np.arange(0.2, 0.7, 0.05)
        pai_range = np.arange(0.2, 0.5, 0.05)
        
        best_f1 = 0.0
        best_thresholds = {}
        
        # Grid search (simplified - only LCS for speed)
        for lcs_thresh in lcs_range:
            # Temporarily set threshold
            original_lcs = self.engine.lcs_threshold
            self.engine.lcs_threshold = lcs_thresh
            
            # Test
            y_true = []
            y_pred = []
            
            for sample in samples:
                block = self.engine.verify_provenance(sample.text)
                y_true.append(1 if sample.true_label == 'authentic' else 0)
                y_pred.append(1 if block.is_sovereign() else 0)
            
            # Calculate F1
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = {
                    'lcs': round(float(lcs_thresh), 3),
                    'truth_pressure': self.engine.truth_threshold,
                    'pai': self.engine.pai_threshold
                }
            
            # Restore
            self.engine.lcs_threshold = original_lcs
        
        print(f"  Best F1: {best_f1:.3f}")
        print(f"  Optimal LCS threshold: {best_thresholds.get('lcs', 0.6)}")
        
        return best_thresholds
    
    # ------------------------------------------------------------------------
    # CROSS-VALIDATION
    # ------------------------------------------------------------------------
    
    def cross_validate(self,
                      samples: List[ValidationSample],
                      n_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            samples: All samples
            n_folds: Number of folds
        
        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn required")
        
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION ({n_folds} folds)")
        print(f"{'='*70}\n")
        
        # Prepare data
        X = list(range(len(samples)))  # Indices
        y = [1 if s.true_label == 'authentic' else 0 for s in samples]
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{n_folds}")
            
            # Test samples for this fold
            test_samples = [samples[i] for i in test_idx]
            
            # Validate
            result = self.validate(test_samples, f"cv_fold_{fold}")
            
            fold_results.append({
                'fold': fold,
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall,
                'accuracy': result.accuracy
            })
        
        # Aggregate results
        avg_f1 = np.mean([r['f1_score'] for r in fold_results])
        std_f1 = np.std([r['f1_score'] for r in fold_results])
        
        cv_results = {
            'n_folds': n_folds,
            'fold_results': fold_results,
            'average_f1': round(avg_f1, 4),
            'std_f1': round(std_f1, 4),
            'average_precision': round(np.mean([r['precision'] for r in fold_results]), 4),
            'average_recall': round(np.mean([r['recall'] for r in fold_results]), 4)
        }
        
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"Average F1-Score: {cv_results['average_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        print(f"Average Precision: {cv_results['average_precision']:.4f}")
        print(f"Average Recall: {cv_results['average_recall']:.4f}")
        print(f"{'='*70}\n")
        
        return cv_results
    
    # ------------------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # ------------------------------------------------------------------------
    
    def _compute_confidence_interval(self,
                                    y_true: List[int],
                                    y_pred: List[int],
                                    n_bootstrap: int = 1000,
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval via bootstrapping"""
        f1_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = [y_true[i] for i in indices]
            y_pred_boot = [y_pred[i] for i in indices]
            
            # Calculate F1
            _, _, f1, _ = precision_recall_fscore_support(
                y_true_boot, y_pred_boot, average='binary', zero_division=0
            )
            f1_scores.append(f1)
        
        # Compute percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(f1_scores, alpha * 100)
        upper = np.percentile(f1_scores, (1 - alpha) * 100)
        
        return (round(float(lower), 4), round(float(upper), 4))
    
    # ------------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------------
    
    def _print_validation_summary(self, result: ValidationResult):
        """Print human-readable validation summary"""
        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}\n")
        
        print(f"Dataset: {result.dataset_name}")
        print(f"Samples: {result.sample_count}")
        print(f"Timestamp: {result.timestamp.isoformat()}\n")
        
        print("CLASSIFICATION METRICS:")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1-Score:  {result.f1_score:.4f}")
        print(f"  Accuracy:  {result.accuracy:.4f}")
        
        if result.confidence_interval:
            ci_low, ci_high = result.confidence_interval
            print(f"  F1 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        if result.auc_score:
            print(f"  AUC-ROC:   {result.auc_score:.4f}")
        
        print("\nCONFUSION MATRIX:")
        print(f"  True Positives:  {result.true_positives}")
        print(f"  True Negatives:  {result.true_negatives}")
        print(f"  False Positives: {result.false_positives}")
        print(f"  False Negatives: {result.false_negatives}")
        
        print("\nCURRENT THRESHOLDS:")
        for key, value in result.current_thresholds.items():
            print(f"  {key}: {value}")
        
        print("\nOPTIMAL THRESHOLDS:")
        for key, value in result.optimal_thresholds.items():
            print(f"  {key}: {value}")
        
        print(f"\n{'='*70}\n")
    
    def export_results(self, filepath: str):
        """Export all validation results to JSON"""
        data = {
            'validation_runs': [r.to_dict() for r in self.validation_history],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported {len(self.validation_history)} validation runs to {filepath}")
    
    def generate_validation_report(self, result: ValidationResult, output_path: str):
        """Generate detailed validation report (text format)"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LYCHEETAH × CASCADE VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {result.dataset_name}\n")
            f.write(f"Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples: {result.sample_count}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Precision:  {result.precision:.4f}\n")
            f.write(f"Recall:     {result.recall:.4f}\n")
            f.write(f"F1-Score:   {result.f1_score:.4f}\n")
            f.write(f"Accuracy:   {result.accuracy:.4f}\n")
            if result.auc_score:
                f.write(f"AUC-ROC:    {result.auc_score:.4f}\n")
            f.write("\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-"*70 + "\n")
            f.write(f"True Positives:  {result.true_positives}\n")
            f.write(f"True Negatives:  {result.true_negatives}\n")
            f.write(f"False Positives: {result.false_positives}\n")
            f.write(f"False Negatives: {result.false_negatives}\n\n")
            
            f.write("DETAILED PREDICTIONS\n")
            f.write("-"*70 + "\n")
            for pred in result.predictions:
                status = "✓" if pred['correct'] else "✗"
                f.write(f"{status} {pred['true_label']:10} -> {pred['predicted_label']:10} ")
                f.write(f"(score: {pred['authenticity_score']:.3f})\n")
                f.write(f"   {pred['text_preview']}\n\n")
        
        print(f"✓ Detailed report saved to {output_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_validation(authentic_dir: str, 
                    generic_dir: str,
                    output_dir: str = "./validation_results") -> ValidationResult:
    """
    Quick validation with sensible defaults
    
    Args:
        authentic_dir: Directory with authentic Lycheetah samples
        generic_dir: Directory with generic samples
        output_dir: Where to save results
    
    Returns:
        Validation results
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize
    engine = ValidationEngine()
    
    # Load data
    samples = engine.load_dataset(authentic_dir, generic_dir)
    
    if len(samples) < 10:
        print("⚠️ WARNING: Very few samples - results may not be reliable")
    
    # Validate
    result = engine.validate(samples, "quick_validation")
    
    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"{output_dir}/validation_{timestamp}.json"
    report_path = f"{output_dir}/validation_report_{timestamp}.txt"
    
    engine.export_results(json_path)
    engine.generate_validation_report(result, report_path)
    
    return result


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LYCHEETAH VALIDATION ENGINE")
    print("="*70 + "\n")
    
    # Create sample validation data
    print("Creating sample validation data...\n")
    
    sample_data = [
        ValidationSample(
            text="The Protector ensures unconditional sacrifice, anchoring defense. The Healer transmutes entropy into truth. The Beacon maintains eternal clarity.",
            true_label='authentic',
            metadata={'type': 'full_trinity'}
        ),
        ValidationSample(
            text="The Protector sacrifices for defense while maintaining guardian protocols.",
            true_label='authentic',
            metadata={'type': 'protector_focus'}
        ),
        ValidationSample(
            text="This system provides security and error correction automatically.",
            true_label='generic',
            metadata={'type': 'generic_technical'}
        ),
        ValidationSample(
            text="The software is fast and reliable with good performance.",
            true_label='generic',
            metadata={'type': 'generic_description'}
        ),
    ]
    
    # Run validation
    validator = ValidationEngine()
    result = validator.validate(sample_data, "demo_validation")
    
    # Export
    validator.export_results("/tmp/validation_demo.json")
    
    print("\n" + "="*70)
    print("✓ VALIDATION ENGINE DEMONSTRATION COMPLETE")
    print("="*70)
