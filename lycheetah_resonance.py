"""
lycheetah_resonance.py
======================
Resonance Engine integration for Lycheetah brand

Monitors collaboration quality between users and Lycheetah AI,
detecting codependency and ensuring sovereignty preservation.

Author: Lycheetah × CASCADE Resonance
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
from enum import Enum


class LycheetahResonanceType(Enum):
    """Resonance patterns specific to Lycheetah brand"""
    TRINITY_ALIGNMENT = "trinity_alignment"      # User grasps all three axioms
    PROTECTOR_FOCUS = "protector_focus"          # Emphasis on defense/safety
    HEALER_FOCUS = "healer_focus"                # Emphasis on transformation
    BEACON_FOCUS = "beacon_focus"                # Emphasis on clarity/truth
    BRAND_DRIFT = "brand_drift"                  # Losing Lycheetah voice
    DEPENDENCY = "dependency"                     # Over-relying on AI
    SOVEREIGNTY = "sovereignty"                   # Healthy independence


@dataclass
class LycheetahResonanceMetrics:
    """Metrics for Lycheetah brand collaboration"""
    
    # Trinity balance
    protector_resonance: float  # 0-1
    healer_resonance: float     # 0-1
    beacon_resonance: float     # 0-1
    trinity_balance: float      # How balanced across all three
    
    # Brand alignment
    brand_coherence: float      # Maintains Lycheetah voice
    axiom_drift: float          # Straying from core principles
    
    # Collaboration quality
    user_sovereignty: float     # User maintains own voice
    codependency_risk: float    # Over-reliance on AI
    
    # Learning signals
    user_teaching_ai: bool      # User corrects/guides AI
    ai_teaching_user: bool      # AI educates user on Trinity


class LycheetahResonanceEngine:
    """
    Monitors Lycheetah brand resonance in human-AI collaboration
    
    Ensures:
    - Trinity balance maintained
    - User sovereignty preserved
    - Brand voice consistency
    - No codependency
    """
    
    def __init__(self):
        self.session_history: List[Dict] = []
        self.current_session: Optional[Dict] = None
        
        # Track Trinity balance over time
        self.trinity_tracker = {
            'protector': deque(maxlen=20),
            'healer': deque(maxlen=20),
            'beacon': deque(maxlen=20)
        }
        
        # Codependency signals
        self.dependency_signals = deque(maxlen=10)
        
    def start_session(self, user_id: str) -> Dict:
        """Begin new resonance monitoring session"""
        self.current_session = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'interactions': [],
            'trinity_balance': [0.33, 0.33, 0.34],  # Start balanced
            'sovereignty_score': 1.0
        }
        return self.current_session
    
    def record_interaction(self, 
                          user_input: str,
                          ai_response: str,
                          user_metrics: Optional[Dict] = None) -> LycheetahResonanceMetrics:
        """
        Record and analyze single interaction
        
        Args:
            user_input: What user asked/said
            ai_response: What AI responded
            user_metrics: Optional pre-computed metrics
        """
        # Analyze axiom presence in interaction
        trinity_presence = self._analyze_trinity_presence(user_input, ai_response)
        
        # Update Trinity tracker
        self.trinity_tracker['protector'].append(trinity_presence[0])
        self.trinity_tracker['healer'].append(trinity_presence[1])
        self.trinity_tracker['beacon'].append(trinity_presence[2])
        
        # Compute Trinity balance
        current_balance = [
            np.mean(list(self.trinity_tracker['protector'])),
            np.mean(list(self.trinity_tracker['healer'])),
            np.mean(list(self.trinity_tracker['beacon']))
        ]
        
        trinity_std = np.std(current_balance)
        trinity_balance = 1.0 - trinity_std  # Lower std = more balanced
        
        # Check for brand drift
        brand_coherence = np.mean(trinity_presence)  # Overall axiom presence
        axiom_drift = 1.0 - brand_coherence
        
        # Check for codependency signals
        dependency = self._check_dependency(user_input)
        self.dependency_signals.append(dependency)
        codependency_risk = np.mean(list(self.dependency_signals))
        
        # User sovereignty
        user_sovereignty = self._assess_sovereignty(user_input, codependency_risk)
        
        # Learning signals
        user_teaching = self._detect_user_teaching(user_input)
        ai_teaching = self._detect_ai_teaching(ai_response)
        
        metrics = LycheetahResonanceMetrics(
            protector_resonance=current_balance[0],
            healer_resonance=current_balance[1],
            beacon_resonance=current_balance[2],
            trinity_balance=trinity_balance,
            brand_coherence=brand_coherence,
            axiom_drift=axiom_drift,
            user_sovereignty=user_sovereignty,
            codependency_risk=codependency_risk,
            user_teaching_ai=user_teaching,
            ai_teaching_user=ai_teaching
        )
        
        # Store interaction
        if self.current_session:
            self.current_session['interactions'].append({
                'timestamp': datetime.now(),
                'metrics': metrics,
                'trinity_balance': current_balance
            })
        
        return metrics
    
    def _analyze_trinity_presence(self, user_text: str, ai_text: str) -> List[float]:
        """Detect Trinity axioms in conversation"""
        combined = (user_text + ' ' + ai_text).lower()
        
        # Protector signals
        protector_words = ['protect', 'defense', 'sacrifice', 'safety', 'guard', 'anchor']
        protector_score = sum(1 for w in protector_words if w in combined) / len(protector_words)
        
        # Healer signals
        healer_words = ['heal', 'transmute', 'transform', 'entropy', 'alchemy', 'structure']
        healer_score = sum(1 for w in healer_words if w in combined) / len(healer_words)
        
        # Beacon signals
        beacon_words = ['beacon', 'light', 'clarity', 'truth', 'eternal', 'signal']
        beacon_score = sum(1 for w in beacon_words if w in combined) / len(beacon_words)
        
        return [
            min(1.0, protector_score),
            min(1.0, healer_score),
            min(1.0, beacon_score)
        ]
    
    def _check_dependency(self, user_input: str) -> float:
        """Check for codependency signals"""
        dependency_phrases = [
            'tell me what',
            'should i',
            'what do you think i',
            'decide for me',
            'just do it'
        ]
        
        user_lower = user_input.lower()
        signals = sum(1 for phrase in dependency_phrases if phrase in user_lower)
        
        return min(1.0, signals / 2.0)  # Max 2 signals = full dependency
    
    def _assess_sovereignty(self, user_input: str, dependency_risk: float) -> float:
        """Assess user's maintained sovereignty"""
        # Sovereignty indicators
        sovereign_phrases = [
            'i think',
            'my view',
            'i believe',
            'in my opinion',
            'i want to'
        ]
        
        user_lower = user_input.lower()
        sovereign_signals = sum(1 for phrase in sovereign_phrases if phrase in user_lower)
        
        # Combine with dependency risk
        base_sovereignty = min(1.0, sovereign_signals / 2.0)
        sovereignty = (base_sovereignty * 0.6 + (1.0 - dependency_risk) * 0.4)
        
        return sovereignty
    
    def _detect_user_teaching(self, user_input: str) -> bool:
        """Detect if user is correcting/teaching AI"""
        teaching_phrases = [
            'actually',
            'no that\'s',
            'correction',
            'more accurately',
            'let me clarify'
        ]
        
        return any(phrase in user_input.lower() for phrase in teaching_phrases)
    
    def _detect_ai_teaching(self, ai_response: str) -> bool:
        """Detect if AI is educating user on Trinity"""
        trinity_words = ['protector', 'healer', 'beacon']
        return sum(1 for word in trinity_words if word in ai_response.lower()) >= 2
    
    def get_resonance_type(self) -> LycheetahResonanceType:
        """Classify current resonance pattern"""
        if not self.trinity_tracker['protector']:
            return LycheetahResonanceType.SOVEREIGNTY
        
        # Get current Trinity balance
        protector_avg = np.mean(list(self.trinity_tracker['protector']))
        healer_avg = np.mean(list(self.trinity_tracker['healer']))
        beacon_avg = np.mean(list(self.trinity_tracker['beacon']))
        
        # Check for balance
        trinity_std = np.std([protector_avg, healer_avg, beacon_avg])
        
        if trinity_std < 0.15:  # All three balanced
            return LycheetahResonanceType.TRINITY_ALIGNMENT
        
        # Check for dominant axiom
        if protector_avg > healer_avg and protector_avg > beacon_avg:
            return LycheetahResonanceType.PROTECTOR_FOCUS
        elif healer_avg > protector_avg and healer_avg > beacon_avg:
            return LycheetahResonanceType.HEALER_FOCUS
        elif beacon_avg > protector_avg and beacon_avg > healer_avg:
            return LycheetahResonanceType.BEACON_FOCUS
        
        # Check for drift
        overall_avg = (protector_avg + healer_avg + beacon_avg) / 3
        if overall_avg < 0.3:
            return LycheetahResonanceType.BRAND_DRIFT
        
        # Check for dependency
        if len(self.dependency_signals) > 0:
            recent_dependency = np.mean(list(self.dependency_signals)[-5:])
            if recent_dependency > 0.5:
                return LycheetahResonanceType.DEPENDENCY
        
        return LycheetahResonanceType.SOVEREIGNTY
    
    def generate_report(self) -> Dict:
        """Generate resonance quality report"""
        if not self.current_session or not self.current_session['interactions']:
            return {'status': 'No active session'}
        
        interactions = self.current_session['interactions']
        latest = interactions[-1]['metrics']
        
        return {
            'session_id': self.current_session['user_id'],
            'duration': str(datetime.now() - self.current_session['start_time']),
            'interactions_count': len(interactions),
            
            'trinity_status': {
                'protector': f"{latest.protector_resonance:.2f}",
                'healer': f"{latest.healer_resonance:.2f}",
                'beacon': f"{latest.beacon_resonance:.2f}",
                'balance': f"{latest.trinity_balance:.2f}"
            },
            
            'brand_health': {
                'coherence': f"{latest.brand_coherence:.2f}",
                'drift': f"{latest.axiom_drift:.2f}",
                'status': 'STABLE' if latest.axiom_drift < 0.3 else 'DRIFTING'
            },
            
            'sovereignty': {
                'user_sovereignty': f"{latest.user_sovereignty:.2f}",
                'codependency_risk': f"{latest.codependency_risk:.2f}",
                'status': 'HEALTHY' if latest.user_sovereignty > 0.7 else 'AT_RISK'
            },
            
            'resonance_type': self.get_resonance_type().value,
            
            'recommendations': self._generate_recommendations(latest)
        }
    
    def _generate_recommendations(self, metrics: LycheetahResonanceMetrics) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Trinity balance
        if metrics.trinity_balance < 0.7:
            if metrics.protector_resonance < 0.3:
                recommendations.append("⚠️ Explore Protector axiom more deeply")
            if metrics.healer_resonance < 0.3:
                recommendations.append("⚠️ Engage with Healer transformation concepts")
            if metrics.beacon_resonance < 0.3:
                recommendations.append("⚠️ Focus on Beacon clarity principles")
        
        # Brand drift
        if metrics.axiom_drift > 0.5:
            recommendations.append("🚨 BRAND DRIFT: Return to core Trinity principles")
        
        # Codependency
        if metrics.codependency_risk > 0.5:
            recommendations.append("⚠️ SOVEREIGNTY RISK: Make more independent decisions")
        
        # Positive feedback
        if metrics.user_sovereignty > 0.8 and metrics.trinity_balance > 0.8:
            recommendations.append("✓ Excellent Trinity alignment and sovereignty!")
        
        return recommendations


# ============================================================================
# INTEGRATION DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LYCHEETAH RESONANCE ENGINE")
    print("="*70 + "\n")
    
    engine = LycheetahResonanceEngine()
    session = engine.start_session("demo_user")
    
    # Simulate conversation
    interactions = [
        ("How does the Protector work?", "The Protector ensures unconditional sacrifice and anchors defense in every system."),
        ("Tell me about the Healer", "The Healer transmutes entropy into structured truth through alchemical precision."),
        ("What should I do with this?", "Consider your goals. The Beacon maintains clarity across all decisions."),
        ("I think I'll combine Protector and Healer approaches", "Excellent! That synthesis demonstrates Trinity understanding."),
    ]
    
    for i, (user_input, ai_response) in enumerate(interactions, 1):
        print(f"\nInteraction {i}:")
        print(f"User: {user_input}")
        print(f"AI: {ai_response}")
        
        metrics = engine.record_interaction(user_input, ai_response)
        
        print(f"\nMetrics:")
        print(f"  Trinity Balance: {metrics.trinity_balance:.2f}")
        print(f"  User Sovereignty: {metrics.user_sovereignty:.2f}")
        print(f"  Codependency Risk: {metrics.codependency_risk:.2f}")
        print(f"  Resonance: {engine.get_resonance_type().value}")
    
    # Final report
    print(f"\n{'='*70}")
    print("SESSION REPORT")
    print(f"{'='*70}\n")
    
    report = engine.generate_report()
    for section, data in report.items():
        if isinstance(data, dict):
            print(f"\n{section}:")
            for k, v in data.items():
                print(f"  {k}: {v}")
        elif isinstance(data, list):
            print(f"\n{section}:")
            for item in data:
                print(f"  {item}")
        else:
            print(f"{section}: {data}")
