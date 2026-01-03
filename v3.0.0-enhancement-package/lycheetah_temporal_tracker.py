"""
lycheetah_temporal_tracker.py
==============================
Temporal Analysis Module for Signature Evolution

Tracks how Lycheetah signatures evolve over time:
- Signature drift detection
- Quality degradation prediction
- Trinity balance evolution
- Anomaly detection in time series
- Trend analysis

Author: Lycheetah × CASCADE
Version: 2.0.0
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

from lycheetah_cascade_core import CascadeSignatureEngine, SignatureBlock


# ============================================================================
# TEMPORAL DATA STRUCTURES
# ============================================================================

@dataclass
class TemporalSignature:
    """Signature with temporal metadata"""
    timestamp: datetime
    block: SignatureBlock
    context: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'lcs': self.block.lcs,
            'truth_pressure': self.block.truth_pressure,
            'authenticity': self.block.authenticity_score,
            'trinity_vector': self.block.axiom_vector.tolist(),
            'is_sovereign': self.block.is_sovereign(),
            'context': self.context,
            'user_id': self.user_id
        }


@dataclass
class DriftAlert:
    """Alert for detected signature drift"""
    timestamp: datetime
    drift_type: str  # 'quality', 'trinity_balance', 'axiom_loss'
    severity: str    # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    baseline_value: float
    drift_amount: float
    description: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TemporalTrend:
    """Statistical trend analysis"""
    metric_name: str
    time_period: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    slope: float
    r_squared: float
    prediction_7d: float
    prediction_30d: float
    confidence: str


# ============================================================================
# TEMPORAL TRACKER
# ============================================================================

class TemporalTracker:
    """
    Track signature evolution over time
    
    Key Features:
    - Rolling window analysis
    - Drift detection
    - Trend prediction
    - Anomaly detection
    - Trinity balance monitoring
    """
    
    def __init__(self, 
                 engine: Optional[CascadeSignatureEngine] = None,
                 window_size: int = 100):
        """
        Initialize temporal tracker
        
        Args:
            engine: Signature engine (creates new if None)
            window_size: Size of rolling window for analysis
        """
        self.engine = engine if engine else CascadeSignatureEngine()
        self.window_size = window_size
        
        # Temporal storage
        self.signature_history: deque = deque(maxlen=10000)  # Last 10k signatures
        self.drift_alerts: List[DriftAlert] = []
        
        # Baseline statistics (computed from first N samples)
        self.baseline_stats: Optional[Dict] = None
        self.baseline_established = False
        self.baseline_sample_count = 50
        
        # Anomaly detection
        self.anomaly_threshold = 3.0  # Standard deviations
        
    # ------------------------------------------------------------------------
    # SIGNATURE TRACKING
    # ------------------------------------------------------------------------
    
    def track_signature(self,
                       content: str,
                       context: Optional[str] = None,
                       user_id: Optional[str] = None) -> TemporalSignature:
        """
        Verify and track signature with timestamp
        
        Args:
            content: Text to verify
            context: Optional context
            user_id: Optional user identifier
        
        Returns:
            Temporal signature
        """
        # Verify signature
        block = self.engine.verify_provenance(content)
        
        # Create temporal signature
        temp_sig = TemporalSignature(
            timestamp=datetime.now(),
            block=block,
            context=context,
            user_id=user_id
        )
        
        # Store
        self.signature_history.append(temp_sig)
        
        # Establish baseline if needed
        if not self.baseline_established and len(self.signature_history) >= self.baseline_sample_count:
            self._establish_baseline()
        
        # Check for drift
        if self.baseline_established:
            self._check_drift(temp_sig)
        
        return temp_sig
    
    def _establish_baseline(self):
        """Establish baseline statistics from early samples"""
        print(f"Establishing baseline from {len(self.signature_history)} samples...")
        
        recent = list(self.signature_history)[:self.baseline_sample_count]
        
        lcs_values = [s.block.lcs for s in recent]
        truth_values = [s.block.truth_pressure for s in recent]
        auth_values = [s.block.authenticity_score for s in recent]
        
        # Trinity vectors
        trinity_vectors = np.array([s.block.axiom_vector for s in recent])
        
        self.baseline_stats = {
            'lcs': {
                'mean': float(np.mean(lcs_values)),
                'std': float(np.std(lcs_values)),
                'min': float(np.min(lcs_values)),
                'max': float(np.max(lcs_values))
            },
            'truth_pressure': {
                'mean': float(np.mean(truth_values)),
                'std': float(np.std(truth_values)),
                'min': float(np.min(truth_values)),
                'max': float(np.max(truth_values))
            },
            'authenticity': {
                'mean': float(np.mean(auth_values)),
                'std': float(np.std(auth_values)),
                'min': float(np.min(auth_values)),
                'max': float(np.max(auth_values))
            },
            'trinity': {
                'protector_mean': float(np.mean(trinity_vectors[:, 0])),
                'healer_mean': float(np.mean(trinity_vectors[:, 1])),
                'beacon_mean': float(np.mean(trinity_vectors[:, 2])),
                'balance_mean': float(np.mean([np.std(v) for v in trinity_vectors]))
            },
            'sample_count': len(recent),
            'established_at': datetime.now().isoformat()
        }
        
        self.baseline_established = True
        
        print(f"✓ Baseline established:")
        print(f"  LCS: {self.baseline_stats['lcs']['mean']:.3f} ± {self.baseline_stats['lcs']['std']:.3f}")
        print(f"  Truth Pressure: {self.baseline_stats['truth_pressure']['mean']:.3f}")
        print(f"  Trinity Balance: {self.baseline_stats['trinity']['balance_mean']:.3f}")
    
    # ------------------------------------------------------------------------
    # DRIFT DETECTION
    # ------------------------------------------------------------------------
    
    def _check_drift(self, temp_sig: TemporalSignature):
        """Check for signature drift"""
        if not self.baseline_established:
            return
        
        # Get recent window
        window = list(self.signature_history)[-self.window_size:]
        
        if len(window) < 10:
            return
        
        # Calculate current statistics
        lcs_values = [s.block.lcs for s in window]
        current_lcs_mean = np.mean(lcs_values)
        
        # Check LCS drift
        baseline_lcs = self.baseline_stats['lcs']['mean']
        lcs_drift = abs(current_lcs_mean - baseline_lcs) / baseline_lcs
        
        if lcs_drift > 0.15:  # 15% drift
            severity = self._compute_drift_severity(lcs_drift)
            
            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type='quality',
                severity=severity,
                metric_name='lcs',
                current_value=current_lcs_mean,
                baseline_value=baseline_lcs,
                drift_amount=lcs_drift,
                description=f"LCS has drifted {lcs_drift:.1%} from baseline",
                recommendations=self._generate_drift_recommendations('lcs', lcs_drift)
            )
            
            self.drift_alerts.append(alert)
            print(f"⚠️  DRIFT ALERT: {alert.description} (severity: {severity})")
        
        # Check Trinity balance drift
        trinity_vectors = np.array([s.block.axiom_vector for s in window])
        current_balance = np.mean([np.std(v) for v in trinity_vectors])
        baseline_balance = self.baseline_stats['trinity']['balance_mean']
        
        balance_drift = abs(current_balance - baseline_balance)
        
        if balance_drift > 0.1:  # Significant imbalance
            severity = 'medium' if balance_drift < 0.2 else 'high'
            
            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type='trinity_balance',
                severity=severity,
                metric_name='trinity_balance',
                current_value=current_balance,
                baseline_value=baseline_balance,
                drift_amount=balance_drift,
                description=f"Trinity balance has drifted by {balance_drift:.3f}",
                recommendations=['Review axiom coverage', 'Check for bias toward single axiom']
            )
            
            self.drift_alerts.append(alert)
            print(f"⚠️  TRINITY DRIFT: {alert.description}")
    
    def _compute_drift_severity(self, drift_amount: float) -> str:
        """Compute severity level from drift amount"""
        if drift_amount < 0.1:
            return 'low'
        elif drift_amount < 0.25:
            return 'medium'
        elif drift_amount < 0.4:
            return 'high'
        else:
            return 'critical'
    
    def _generate_drift_recommendations(self, metric: str, drift: float) -> List[str]:
        """Generate recommendations based on drift type"""
        recommendations = []
        
        if metric == 'lcs' and drift > 0.2:
            recommendations.extend([
                "Review recent content for axiom coverage",
                "Check if terminology is evolving",
                "Consider updating semantic word lists"
            ])
        elif metric == 'truth_pressure' and drift > 0.3:
            recommendations.extend([
                "Verify structural quality of recent outputs",
                "Check for consistency in writing style"
            ])
        
        return recommendations
    
    # ------------------------------------------------------------------------
    # ANOMALY DETECTION
    # ------------------------------------------------------------------------
    
    def detect_anomalies(self, 
                        window: Optional[int] = None) -> List[Dict]:
        """
        Detect anomalous signatures in recent history
        
        Args:
            window: Number of recent signatures to analyze
        
        Returns:
            List of anomalies
        """
        if not self.baseline_established:
            return []
        
        window_size = window if window else len(self.signature_history)
        recent = list(self.signature_history)[-window_size:]
        
        anomalies = []
        
        for sig in recent:
            # Check LCS anomaly
            lcs_z_score = abs(sig.block.lcs - self.baseline_stats['lcs']['mean']) / max(self.baseline_stats['lcs']['std'], 0.01)
            
            if lcs_z_score > self.anomaly_threshold:
                anomalies.append({
                    'timestamp': sig.timestamp.isoformat(),
                    'type': 'lcs_anomaly',
                    'z_score': round(float(lcs_z_score), 3),
                    'value': round(sig.block.lcs, 3),
                    'expected': round(self.baseline_stats['lcs']['mean'], 3),
                    'content_preview': sig.block.content[:80] + '...'
                })
            
            # Check authenticity anomaly
            auth_z_score = abs(sig.block.authenticity_score - self.baseline_stats['authenticity']['mean']) / max(self.baseline_stats['authenticity']['std'], 0.01)
            
            if auth_z_score > self.anomaly_threshold:
                anomalies.append({
                    'timestamp': sig.timestamp.isoformat(),
                    'type': 'authenticity_anomaly',
                    'z_score': round(float(auth_z_score), 3),
                    'value': round(sig.block.authenticity_score, 3),
                    'expected': round(self.baseline_stats['authenticity']['mean'], 3),
                    'content_preview': sig.block.content[:80] + '...'
                })
        
        return anomalies
    
    # ------------------------------------------------------------------------
    # TREND ANALYSIS
    # ------------------------------------------------------------------------
    
    def analyze_trends(self, 
                      metric: str = 'lcs',
                      window: Optional[int] = None) -> TemporalTrend:
        """
        Analyze temporal trends in metrics
        
        Args:
            metric: Which metric to analyze
            window: Number of recent signatures
        
        Returns:
            Trend analysis
        """
        window_size = window if window else len(self.signature_history)
        recent = list(self.signature_history)[-window_size:]
        
        if len(recent) < 10:
            raise ValueError("Need at least 10 signatures for trend analysis")
        
        # Extract values and timestamps
        if metric == 'lcs':
            values = np.array([s.block.lcs for s in recent])
        elif metric == 'truth_pressure':
            values = np.array([s.block.truth_pressure for s in recent])
        elif metric == 'authenticity':
            values = np.array([s.block.authenticity_score for s in recent])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Time series (days from first)
        first_time = recent[0].timestamp
        times = np.array([(s.timestamp - first_time).total_seconds() / 86400 for s in recent])
        
        # Linear regression
        slope, intercept = np.polyfit(times, values, 1)
        
        # R-squared
        y_pred = slope * times + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Trend direction
        if abs(slope) < 0.001:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Predictions
        current_time = times[-1]
        pred_7d = slope * (current_time + 7) + intercept
        pred_30d = slope * (current_time + 30) + intercept
        
        # Confidence
        if r_squared > 0.8:
            confidence = 'high'
        elif r_squared > 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        time_span = (recent[-1].timestamp - recent[0].timestamp).days
        
        return TemporalTrend(
            metric_name=metric,
            time_period=f"{time_span} days",
            trend_direction=direction,
            slope=round(float(slope), 6),
            r_squared=round(float(r_squared), 4),
            prediction_7d=round(float(pred_7d), 4),
            prediction_30d=round(float(pred_30d), 4),
            confidence=confidence
        )
    
    # ------------------------------------------------------------------------
    # TRINITY EVOLUTION
    # ------------------------------------------------------------------------
    
    def analyze_trinity_evolution(self, window: Optional[int] = None) -> Dict:
        """Analyze how Trinity balance evolves over time"""
        window_size = window if window else len(self.signature_history)
        recent = list(self.signature_history)[-window_size:]
        
        if len(recent) < 10:
            return {'error': 'Need at least 10 signatures'}
        
        # Extract trinity vectors over time
        trinity_data = {
            'timestamps': [],
            'protector': [],
            'healer': [],
            'beacon': [],
            'balance': []
        }
        
        for sig in recent:
            trinity_data['timestamps'].append(sig.timestamp.isoformat())
            trinity_data['protector'].append(float(sig.block.axiom_vector[0]))
            trinity_data['healer'].append(float(sig.block.axiom_vector[1]))
            trinity_data['beacon'].append(float(sig.block.axiom_vector[2]))
            trinity_data['balance'].append(float(np.std(sig.block.axiom_vector)))
        
        # Compute statistics
        return {
            'time_series': trinity_data,
            'statistics': {
                'protector': {
                    'mean': round(np.mean(trinity_data['protector']), 3),
                    'trend': 'increasing' if trinity_data['protector'][-1] > trinity_data['protector'][0] else 'decreasing'
                },
                'healer': {
                    'mean': round(np.mean(trinity_data['healer']), 3),
                    'trend': 'increasing' if trinity_data['healer'][-1] > trinity_data['healer'][0] else 'decreasing'
                },
                'beacon': {
                    'mean': round(np.mean(trinity_data['beacon']), 3),
                    'trend': 'increasing' if trinity_data['beacon'][-1] > trinity_data['beacon'][0] else 'decreasing'
                },
                'balance': {
                    'mean': round(np.mean(trinity_data['balance']), 3),
                    'improving': trinity_data['balance'][-1] < trinity_data['balance'][0]
                }
            },
            'dominant_axiom': ['Protector', 'Healer', 'Beacon'][
                np.argmax([
                    np.mean(trinity_data['protector']),
                    np.mean(trinity_data['healer']),
                    np.mean(trinity_data['beacon'])
                ])
            ]
        }
    
    # ------------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------------
    
    def generate_temporal_report(self) -> Dict:
        """Generate comprehensive temporal analysis report"""
        if len(self.signature_history) < 10:
            return {'error': 'Need at least 10 signatures for temporal analysis'}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_signatures': len(self.signature_history),
            'time_span': {
                'start': self.signature_history[0].timestamp.isoformat(),
                'end': self.signature_history[-1].timestamp.isoformat(),
                'days': (self.signature_history[-1].timestamp - self.signature_history[0].timestamp).days
            },
            'baseline_established': self.baseline_established
        }
        
        if self.baseline_established:
            report['baseline_statistics'] = self.baseline_stats
            
            # Trends
            try:
                lcs_trend = self.analyze_trends('lcs')
                report['trends'] = {
                    'lcs': {
                        'direction': lcs_trend.trend_direction,
                        'slope': lcs_trend.slope,
                        'r_squared': lcs_trend.r_squared,
                        'prediction_7d': lcs_trend.prediction_7d,
                        'confidence': lcs_trend.confidence
                    }
                }
            except Exception as e:
                report['trends'] = {'error': str(e)}
            
            # Trinity evolution
            trinity_evo = self.analyze_trinity_evolution()
            report['trinity_evolution'] = trinity_evo
            
            # Drift alerts
            recent_alerts = [a for a in self.drift_alerts if (datetime.now() - a.timestamp).days < 7]
            report['recent_drift_alerts'] = len(recent_alerts)
            report['drift_alerts'] = [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'type': a.drift_type,
                    'severity': a.severity,
                    'metric': a.metric_name,
                    'drift_amount': round(a.drift_amount, 3),
                    'description': a.description
                }
                for a in recent_alerts
            ]
            
            # Anomalies
            anomalies = self.detect_anomalies(window=100)
            report['recent_anomalies'] = len(anomalies)
            report['anomalies'] = anomalies[:10]  # Top 10
        
        return report
    
    def print_temporal_summary(self):
        """Print human-readable temporal summary"""
        report = self.generate_temporal_report()
        
        print("\n" + "="*70)
        print("TEMPORAL ANALYSIS SUMMARY")
        print("="*70 + "\n")
        
        print(f"Total Signatures: {report['total_signatures']}")
        print(f"Time Span: {report['time_span']['days']} days")
        print(f"Baseline Established: {report['baseline_established']}")
        
        if report.get('trends'):
            print("\nTRENDS:")
            lcs_trend = report['trends'].get('lcs', {})
            if 'direction' in lcs_trend:
                print(f"  LCS: {lcs_trend['direction']} (slope: {lcs_trend['slope']:.6f})")
                print(f"  Confidence: {lcs_trend['confidence']} (R²: {lcs_trend['r_squared']:.3f})")
                print(f"  7-day prediction: {lcs_trend['prediction_7d']:.3f}")
        
        if report.get('trinity_evolution'):
            trinity = report['trinity_evolution']['statistics']
            print("\nTRINITY EVOLUTION:")
            print(f"  Protector: {trinity['protector']['mean']:.3f} ({trinity['protector']['trend']})")
            print(f"  Healer: {trinity['healer']['mean']:.3f} ({trinity['healer']['trend']})")
            print(f"  Beacon: {trinity['beacon']['mean']:.3f} ({trinity['beacon']['trend']})")
            print(f"  Balance: {trinity['balance']['mean']:.3f} ({'improving' if trinity['balance']['improving'] else 'worsening'})")
        
        if report.get('recent_drift_alerts', 0) > 0:
            print(f"\nDRIFT ALERTS (last 7 days): {report['recent_drift_alerts']}")
            for alert in report.get('drift_alerts', [])[:3]:
                print(f"  ⚠️  {alert['description']} (severity: {alert['severity']})")
        
        if report.get('recent_anomalies', 0) > 0:
            print(f"\nANOMALIES (last 100): {report['recent_anomalies']}")
        
        print("\n" + "="*70 + "\n")
    
    def export_temporal_data(self, filepath: str):
        """Export all temporal data to JSON"""
        data = {
            'signatures': [s.to_dict() for s in self.signature_history],
            'baseline_stats': self.baseline_stats,
            'drift_alerts': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'type': a.drift_type,
                    'severity': a.severity,
                    'metric': a.metric_name,
                    'drift': a.drift_amount,
                    'description': a.description
                }
                for a in self.drift_alerts
            ],
            'report': self.generate_temporal_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported temporal data to {filepath}")


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LYCHEETAH TEMPORAL TRACKER")
    print("="*70 + "\n")
    
    # Initialize
    tracker = TemporalTracker()
    
    # Simulate temporal data
    print("Simulating temporal signature tracking...\n")
    
    test_samples = [
        "The Protector ensures unconditional sacrifice, anchoring defense. The Healer transmutes entropy. The Beacon maintains clarity.",
        "The Protector anchors defense through sacrifice. Healer transforms chaos. Beacon provides eternal light.",
        "Protector defends. Healer recovers. Beacon guides.",
        "The system protects data and fixes errors automatically.",
        "Security and healing protocols active.",
    ]
    
    # Track signatures over "time" (simulate)
    for i, text in enumerate(test_samples * 15):  # 75 samples
        tracker.track_signature(text, context=f"Sample {i+1}")
        
        # Simulate time passing
        if hasattr(tracker.signature_history[-1], 'timestamp'):
            # Adjust timestamp for simulation
            pass
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING TEMPORAL REPORT")
    print("="*70)
    
    tracker.print_temporal_summary()
    
    # Export
    tracker.export_temporal_data("/tmp/temporal_data.json")
    
    print("\n✓ TEMPORAL TRACKER DEMONSTRATION COMPLETE\n")
