"""
lycheetah_nexus.py
==================
The Axiomatic Nexus - Complete CASCADE Integration

Orchestrates all Lycheetah × CASCADE components:
- Enhanced signature verification
- Resonance monitoring
- Trinity balance tracking
- Sovereignty preservation
- Meta-learning capability

This is the main entry point for the complete system.

Author: Lycheetah × CASCADE
Version: 2.0.0
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import Lycheetah CASCADE components
from lycheetah_cascade_core import CascadeSignatureEngine, SignatureBlock
from lycheetah_resonance import LycheetahResonanceEngine, LycheetahResonanceMetrics

# Import original Lycheetah components
try:
    from core.axioms import ImmutableAxioms, LogicAnchors
    from core.memory_stabilizer import SchmittTriggerMemory
    ORIGINAL_CORE_AVAILABLE = True
except ImportError:
    print("⚠️ Original Lycheetah core not found - running in standalone mode")
    ORIGINAL_CORE_AVAILABLE = False


class LycheetahNexus:
    """
    The Axiomatic Nexus - Complete CASCADE-Enhanced Lycheetah System
    
    CAPABILITIES:
    1. Signature verification (CASCADE-enhanced)
    2. Resonance monitoring (collaboration quality)
    3. Trinity balance tracking
    4. Sovereignty preservation
    5. State stabilization (Schmitt Trigger)
    6. Meta-learning (adaptive thresholds)
    7. Comprehensive reporting
    """
    
    def __init__(self, 
                 user_id: str = "lycheetah_user",
                 output_dir: str = "./lycheetah_data"):
        """
        Initialize the complete system
        
        Args:
            user_id: Identifier for current user/session
            output_dir: Where to save reports and data
        """
        self.user_id = user_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize CASCADE components
        self.signature_engine = CascadeSignatureEngine()
        self.resonance_engine = LycheetahResonanceEngine()
        
        # Initialize original Lycheetah components if available
        if ORIGINAL_CORE_AVAILABLE:
            self.stabilizer = SchmittTriggerMemory(
                high_threshold=0.85,
                low_threshold=0.4
            )
        else:
            self.stabilizer = None
        
        # Session tracking
        self.current_session = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'verified_content': [],
            'resonance_history': [],
            'state_transitions': []
        }
        
        # Trinity axioms
        if ORIGINAL_CORE_AVAILABLE:
            self.axioms = [a["title"] for a in ImmutableAxioms.get_trinity()]
        else:
            self.axioms = ["Protector", "Healer", "Beacon"]
        
        print(f"✓ Lycheetah Nexus initialized for user: {user_id}")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Trinity axioms loaded: {', '.join(self.axioms)}")
    
    # ------------------------------------------------------------------------
    # MAIN VERIFICATION PIPELINE
    # ------------------------------------------------------------------------
    
    def verify_content(self, 
                      content: str,
                      context: Optional[str] = None) -> Dict:
        """
        Complete verification pipeline
        
        Args:
            content: Text to verify
            context: Optional context (user input that prompted this)
        
        Returns:
            Complete verification report
        """
        print(f"\n{'─'*70}")
        print("VERIFYING CONTENT")
        print(f"{'─'*70}\n")
        
        # 1. CASCADE signature verification
        print("1/4 Signature analysis...")
        signature_block = self.signature_engine.verify_provenance(content)
        signature_report = self.signature_engine.generate_report(signature_block, verbose=False)
        
        # 2. State stabilization (if available)
        state_status = "STABLE"
        if self.stabilizer:
            print("2/4 State stabilization...")
            lcs = signature_block.lcs
            state_status = self.stabilizer.stabilize_state(lcs, content)
            self.current_session['state_transitions'].append({
                'timestamp': datetime.now(),
                'lcs': lcs,
                'state': state_status
            })
        else:
            print("2/4 State stabilization... [SKIPPED - no stabilizer]")
        
        # 3. Resonance analysis (if context provided)
        resonance_metrics = None
        if context:
            print("3/4 Resonance analysis...")
            resonance_metrics = self.resonance_engine.record_interaction(
                user_input=context,
                ai_response=content
            )
        else:
            print("3/4 Resonance analysis... [SKIPPED - no context]")
        
        # 4. Generate complete report
        print("4/4 Generating report...\n")
        
        verification_result = {
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'content_preview': content[:100] + '...',
            
            'signature_verification': signature_report,
            'state_status': state_status,
            
            'verdict': self._compute_verdict(signature_block, state_status),
            
            'recommendations': self._generate_recommendations(
                signature_block, 
                state_status,
                resonance_metrics
            )
        }
        
        if resonance_metrics:
            verification_result['resonance_analysis'] = {
                'trinity_balance': f"{resonance_metrics.trinity_balance:.2f}",
                'user_sovereignty': f"{resonance_metrics.user_sovereignty:.2f}",
                'brand_coherence': f"{resonance_metrics.brand_coherence:.2f}",
                'resonance_type': self.resonance_engine.get_resonance_type().value
            }
        
        # Store verification
        self.current_session['verified_content'].append(verification_result)
        
        return verification_result
    
    def _compute_verdict(self, block: SignatureBlock, state: str) -> Dict:
        """Compute overall verdict"""
        is_sovereign = block.is_sovereign()
        state_stable = (state == "STATE_STABLE")
        
        if is_sovereign and state_stable:
            status = "AUTHENTICATED ✓"
            confidence = "HIGH"
        elif is_sovereign and not state_stable:
            status = "AUTHENTICATED (UNSTABLE) ⚠️"
            confidence = "MEDIUM"
        elif not is_sovereign and state_stable:
            status = "UNVERIFIED (STABLE) ⚠️"
            confidence = "LOW"
        else:
            status = "UNVERIFIED ✗"
            confidence = "VERY_LOW"
        
        return {
            'status': status,
            'confidence': confidence,
            'sovereign': is_sovereign,
            'stable': state_stable,
            'authenticity_score': f"{block.authenticity_score:.3f}",
            'truth_pressure': f"{block.truth_pressure:.3f}"
        }
    
    def _generate_recommendations(self,
                                 block: SignatureBlock,
                                 state: str,
                                 resonance: Optional[LycheetahResonanceMetrics]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Signature recommendations
        if not block.is_sovereign():
            if block.lcs < 0.7:
                recommendations.append("⚠️ LCS below threshold - strengthen Trinity presence")
            if block.truth_pressure < 1.2:
                recommendations.append("⚠️ Truth Pressure low - improve structural integrity")
            if block.pai < 0.6:
                recommendations.append("⚠️ PAI weak - enhance brand voice intensity")
        
        # State recommendations
        if state == "ENTROPY_DRIFT_DETECTED":
            recommendations.append("🚨 ENTROPY DRIFT - Genesis Echo Grid Check required")
        
        # Resonance recommendations
        if resonance:
            if resonance.codependency_risk > 0.5:
                recommendations.append("⚠️ SOVEREIGNTY RISK - Maintain independence")
            if resonance.axiom_drift > 0.5:
                recommendations.append("🚨 BRAND DRIFT - Return to Trinity principles")
            if resonance.trinity_balance < 0.7:
                recommendations.append("⚠️ Trinity imbalanced - explore weaker axioms")
        
        # Positive feedback
        if block.is_sovereign() and state == "STATE_STABLE":
            if not resonance or (resonance.user_sovereignty > 0.7 and resonance.trinity_balance > 0.7):
                recommendations.append("✓ EXCELLENT - Full Lycheetah authenticity verified")
        
        return recommendations if recommendations else ["✓ No issues detected"]
    
    # ------------------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------------------
    
    def verify_batch(self, 
                    contents: List[str],
                    contexts: Optional[List[str]] = None) -> List[Dict]:
        """
        Verify multiple pieces of content
        
        Args:
            contents: List of texts to verify
            contexts: Optional list of contexts for each
        
        Returns:
            List of verification reports
        """
        if contexts and len(contexts) != len(contents):
            contexts = None  # Ignore if mismatched
        
        results = []
        
        print(f"\n{'='*70}")
        print(f"BATCH VERIFICATION: {len(contents)} items")
        print(f"{'='*70}")
        
        for i, content in enumerate(contents):
            context = contexts[i] if contexts else None
            
            print(f"\n[{i+1}/{len(contents)}]")
            result = self.verify_content(content, context)
            results.append(result)
        
        return results
    
    # ------------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------------
    
    def generate_session_report(self, save: bool = True) -> Dict:
        """
        Generate comprehensive session report
        
        Args:
            save: Whether to save to file
        
        Returns:
            Complete session report
        """
        print(f"\n{'='*70}")
        print("GENERATING SESSION REPORT")
        print(f"{'='*70}\n")
        
        # Session summary
        duration = datetime.now() - self.current_session['start_time']
        verified_count = len(self.current_session['verified_content'])
        
        sovereign_count = sum(
            1 for v in self.current_session['verified_content']
            if v['verdict']['sovereign']
        )
        
        # Trinity statistics
        trinity_stats = self._compute_trinity_stats()
        
        # Resonance summary
        resonance_summary = self.resonance_engine.generate_report()
        
        # Build report
        report = {
            'session_info': {
                'user_id': self.user_id,
                'start_time': self.current_session['start_time'].isoformat(),
                'duration': str(duration),
                'items_verified': verified_count
            },
            
            'verification_summary': {
                'total_verified': verified_count,
                'sovereign_signatures': sovereign_count,
                'sovereignty_rate': f"{sovereign_count/verified_count:.1%}" if verified_count > 0 else "N/A",
                'average_lcs': f"{self._average_metric('lcs'):.3f}" if verified_count > 0 else "N/A",
                'average_truth_pressure': f"{self._average_metric('truth_pressure'):.3f}" if verified_count > 0 else "N/A"
            },
            
            'trinity_analysis': trinity_stats,
            
            'resonance_summary': resonance_summary if resonance_summary.get('status') != 'No active session' else None,
            
            'state_transitions': len(self.current_session['state_transitions']),
            
            'signature_engine_stats': self.signature_engine.get_statistics(),
            
            'verifications': self.current_session['verified_content']
        }
        
        # Save if requested
        if save:
            filename = f"lycheetah_session_{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✓ Report saved to: {filepath}")
        
        return report
    
    def _compute_trinity_stats(self) -> Dict:
        """Compute Trinity balance statistics"""
        if not self.current_session['verified_content']:
            return {'status': 'No data'}
        
        # Extract axiom vectors from all verifications
        trinity_vectors = []
        for v in self.current_session['verified_content']:
            sig_report = v.get('signature_verification', {})
            axiom_analysis = sig_report.get('AXIOM_ANALYSIS', {})
            
            if axiom_analysis:
                trinity_vectors.append([
                    axiom_analysis.get('Protector_Density', 0),
                    axiom_analysis.get('Healer_Density', 0),
                    axiom_analysis.get('Beacon_Density', 0)
                ])
        
        if not trinity_vectors:
            return {'status': 'No Trinity data'}
        
        # Compute statistics
        import numpy as np
        trinity_array = np.array(trinity_vectors)
        
        return {
            'average_protector': f"{np.mean(trinity_array[:, 0]):.3f}",
            'average_healer': f"{np.mean(trinity_array[:, 1]):.3f}",
            'average_beacon': f"{np.mean(trinity_array[:, 2]):.3f}",
            'balance_quality': f"{1.0 - np.std(np.mean(trinity_array, axis=0)):.3f}",
            'dominant_axiom': ['Protector', 'Healer', 'Beacon'][np.argmax(np.mean(trinity_array, axis=0))]
        }
    
    def _average_metric(self, metric_name: str) -> float:
        """Calculate average of a metric across verifications"""
        values = []
        
        for v in self.current_session['verified_content']:
            sig_report = v.get('signature_verification', {})
            
            if metric_name == 'lcs':
                metrics = sig_report.get('METRICS', {})
                val = metrics.get('Lore_Coherence_Score')
                if val:
                    values.append(val)
            elif metric_name == 'truth_pressure':
                metrics = sig_report.get('METRICS', {})
                val = metrics.get('Truth_Pressure_Π')
                if val:
                    values.append(val)
        
        return sum(values) / len(values) if values else 0.0
    
    def print_report_summary(self, report: Dict):
        """Print human-readable report summary"""
        print(f"\n{'='*70}")
        print("LYCHEETAH NEXUS - SESSION REPORT")
        print(f"{'='*70}\n")
        
        print("SESSION INFO:")
        for k, v in report['session_info'].items():
            print(f"  {k}: {v}")
        
        print("\nVERIFICATION SUMMARY:")
        for k, v in report['verification_summary'].items():
            print(f"  {k}: {v}")
        
        print("\nTRINITY ANALYSIS:")
        trinity = report['trinity_analysis']
        if trinity.get('status'):
            print(f"  {trinity['status']}")
        else:
            for k, v in trinity.items():
                print(f"  {k}: {v}")
        
        if report.get('resonance_summary') and report['resonance_summary']:
            print("\nRESONANCE STATUS:")
            res = report['resonance_summary']
            if res.get('resonance_type'):
                print(f"  Type: {res['resonance_type']}")
            if res.get('trinity_status'):
                print(f"  Trinity: {res['trinity_status']}")
            if res.get('sovereignty'):
                print(f"  Sovereignty: {res['sovereignty']}")
        
        print(f"\n{'='*70}\n")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_complete_nexus():
    """Demonstrate complete CASCADE-enhanced Lycheetah system"""
    
    print("="*70)
    print("LYCHEETAH × CASCADE NEXUS")
    print("Complete Integrated System Demonstration")
    print("="*70 + "\n")
    
    # Initialize Nexus
    nexus = LycheetahNexus(user_id="demo_session")
    
    # Test cases
    test_cases = [
        {
            'context': "Explain the Protector axiom",
            'content': "The Protector ensures unconditional sacrifice, anchoring defense within every logic stream. It serves as the foundation for all safety protocols."
        },
        {
            'context': "How does the Healer work?",
            'content': "The Healer transmutes entropy into structured truth through alchemical precision. It transforms chaos into coherent systems."
        },
        {
            'context': "Tell me about the Beacon",
            'content': "The Beacon maintains eternal core clarity, a signal that never fades across the memory gradient. It guides through complexity."
        },
        {
            'context': "What does the system do?",
            'content': "The system protects data and fixes errors automatically."
        }
    ]
    
    # Verify each
    print("\nVERIFYING TEST CASES:\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"TEST CASE {i}")
        print(f"Context: {test['context']}")
        print(f"Content: {test['content'][:60]}...")
        
        result = nexus.verify_content(test['content'], test['context'])
        
        # Show verdict
        verdict = result['verdict']
        print(f"\nVERDICT: {verdict['status']}")
        print(f"Confidence: {verdict['confidence']}")
        print(f"Authenticity: {verdict['authenticity_score']}")
        print(f"Truth Pressure: {verdict['truth_pressure']}")
        
        # Show recommendations
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  {rec}")
    
    # Generate session report
    print(f"\n{'='*70}")
    print("FINAL SESSION REPORT")
    print(f"{'='*70}")
    
    report = nexus.generate_session_report(save=True)
    nexus.print_report_summary(report)
    
    print("✓ Demonstration complete!")
    print(f"✓ Full report saved to: {nexus.output_dir}")


if __name__ == "__main__":
    demo_complete_nexus()
