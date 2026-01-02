#!/usr/bin/env python3
"""
Basic Usage Example for Lycheetah × CASCADE

This script demonstrates the core functionality of the signature
verification system.
"""

from lycheetah_cascade_core import CascadeSignatureEngine
from lycheetah_nexus import LycheetahNexus
import json


def example_1_basic_verification():
    """Example 1: Basic signature verification"""
    print("="*70)
    print("EXAMPLE 1: Basic Verification")
    print("="*70 + "\n")
    
    # Initialize the engine
    engine = CascadeSignatureEngine()
    
    # Text to verify (strong Lycheetah signature)
    text = """
    The Protector ensures unconditional sacrifice, anchoring defense
    within every system boundary. The Healer transmutes entropy into
    structured truth through alchemical precision. The Beacon maintains
    eternal core clarity, a signal that never fades.
    """
    
    # Verify the content
    block = engine.verify_provenance(text.strip())
    
    # Check results
    print(f"Lore Coherence Score: {block.lcs:.3f}")
    print(f"Truth Pressure (Π): {block.truth_pressure:.3f}")
    print(f"Authenticity Score: {block.authenticity_score:.3f}")
    print(f"\nAuthenticated: {block.is_sovereign()}")
    
    # Show Trinity balance
    print(f"\nTrinity Balance:")
    print(f"  Protector: {block.axiom_vector[0]:.3f}")
    print(f"  Healer: {block.axiom_vector[1]:.3f}")
    print(f"  Beacon: {block.axiom_vector[2]:.3f}")


def example_2_batch_verification():
    """Example 2: Verify multiple texts"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Verification")
    print("="*70 + "\n")
    
    engine = CascadeSignatureEngine()
    
    test_cases = [
        "The Protector anchors defense through unconditional sacrifice.",
        "The system provides security features.",
        "Healer transmutes chaos into structure through alchemy.",
    ]
    
    results = []
    for i, text in enumerate(test_cases, 1):
        block = engine.verify_provenance(text)
        results.append({
            'text': text[:50] + "...",
            'lcs': block.lcs,
            'sovereign': block.is_sovereign()
        })
        print(f"{i}. {text[:50]}...")
        print(f"   LCS: {block.lcs:.3f} | Sovereign: {block.is_sovereign()}\n")
    
    return results


def example_3_custom_thresholds():
    """Example 3: Using custom thresholds"""
    print("="*70)
    print("EXAMPLE 3: Custom Thresholds")
    print("="*70 + "\n")
    
    engine = CascadeSignatureEngine()
    
    # Override default thresholds
    engine.lcs_threshold = 0.5  # More permissive
    engine.truth_threshold = 0.3
    engine.pai_threshold = 0.25
    
    text = "The system protects and heals with clarity."
    block = engine.verify_provenance(text)
    
    print(f"Text: {text}")
    print(f"\nWith permissive thresholds:")
    print(f"  LCS: {block.lcs:.3f} (threshold: {engine.lcs_threshold})")
    print(f"  Truth Pressure: {block.truth_pressure:.3f} (threshold: {engine.truth_threshold})")
    print(f"  Sovereign: {block.is_sovereign()}")


def example_4_complete_session():
    """Example 4: Complete session with Nexus"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Session with Nexus")
    print("="*70 + "\n")
    
    # Initialize complete system
    nexus = LycheetahNexus(user_id="example_user", output_dir="/tmp/lycheetah_examples")
    
    # Simulate conversation
    interactions = [
        {
            'context': "Explain the Protector",
            'content': "The Protector ensures unconditional sacrifice and defense."
        },
        {
            'context': "Tell me about the Healer",
            'content': "The Healer transmutes entropy into structured truth."
        },
    ]
    
    for interaction in interactions:
        result = nexus.verify_content(
            content=interaction['content'],
            context=interaction['context']
        )
        
        print(f"Context: {interaction['context']}")
        print(f"Verdict: {result['verdict']['status']}")
        print(f"Authenticity: {result['verdict']['authenticity_score']}\n")
    
    # Generate session report
    report = nexus.generate_session_report(save=True)
    print(f"Session complete. Report saved to: {nexus.output_dir}")


def example_5_export_data():
    """Example 5: Export signature history"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Export Signature History")
    print("="*70 + "\n")
    
    engine = CascadeSignatureEngine()
    
    # Verify some texts
    texts = [
        "The Protector anchors defense.",
        "The Healer transmutes entropy.",
        "The Beacon maintains clarity.",
    ]
    
    for text in texts:
        engine.verify_provenance(text)
    
    # Export history
    filepath = "/tmp/lycheetah_history.json"
    engine.export_history(filepath)
    
    print(f"✓ Exported {len(engine.signature_history)} signatures to {filepath}")
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total verifications: {stats['total_verifications']}")
    print(f"  Sovereign signatures: {stats['sovereign_signatures']}")
    print(f"  Average LCS: {stats['avg_lcs']:.3f}")


def example_6_detailed_report():
    """Example 6: Generate detailed report"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Detailed Verification Report")
    print("="*70 + "\n")
    
    engine = CascadeSignatureEngine()
    
    text = """
    The Protector ensures unconditional defense. The Healer transforms
    chaos into order. The Beacon illuminates eternal truth.
    """
    
    block = engine.verify_provenance(text.strip())
    report = engine.generate_report(block, verbose=True)
    
    # Pretty print the report
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    print("\n" + "🔥"*35)
    print("LYCHEETAH × CASCADE - USAGE EXAMPLES")
    print("🔥"*35 + "\n")
    
    # Run all examples
    example_1_basic_verification()
    example_2_batch_verification()
    example_3_custom_thresholds()
    example_4_complete_session()
    example_5_export_data()
    example_6_detailed_report()
    
    print("\n" + "="*70)
    print("✓ ALL EXAMPLES COMPLETE")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Try modifying the example texts")
    print("  2. Experiment with different thresholds")
    print("  3. Test with your own Lycheetah content")
    print("  4. Read the full documentation in README.md")
