#!/usr/bin/env python3
"""
lycheetah_cli.py
================
Command-line interface for Lycheetah × CASCADE system

Simple tool to verify content signatures from the command line.

Usage:
    python lycheetah_cli.py verify "Your text here"
    python lycheetah_cli.py verify-file document.txt
    python lycheetah_cli.py session
    
Author: Lycheetah × CASCADE
Version: 2.0.1
"""

import sys
import argparse
from pathlib import Path

# Import Lycheetah components
try:
    from lycheetah_cascade_core import CascadeSignatureEngine
    from lycheetah_resonance import LycheetahResonanceEngine
    from lycheetah_nexus import LycheetahNexus
    from lycheetah_config import SignatureThresholds, SystemConfig
except ImportError as e:
    print(f"Error: Could not import Lycheetah modules: {e}")
    print("Make sure all Lycheetah files are in the same directory.")
    sys.exit(1)


def verify_text(text: str, verbose: bool = False):
    """Verify a single piece of text"""
    print("="*70)
    print("LYCHEETAH SIGNATURE VERIFICATION")
    print("="*70 + "\n")
    
    engine = CascadeSignatureEngine()
    block = engine.verify_provenance(text)
    report = engine.generate_report(block, verbose=verbose)
    
    # Print report
    for section, data in report.items():
        if isinstance(data, dict):
            print(f"\n{section}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {data}")
    
    print(f"\n{'='*70}")
    
    # Simple verdict
    if block.is_sovereign():
        print("✓ AUTHENTICATED - Lycheetah signature detected")
    else:
        print("✗ UNVERIFIED - Does not meet signature thresholds")
    
    print(f"{'='*70}\n")


def verify_file(filepath: str, verbose: bool = False):
    """Verify content from a file"""
    path = Path(filepath)
    
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    print(f"Verifying file: {filepath}")
    print(f"Size: {len(text)} characters\n")
    
    verify_text(text, verbose)


def start_session(user_id: str = "cli_user"):
    """Start an interactive verification session"""
    print("="*70)
    print("LYCHEETAH INTERACTIVE SESSION")
    print("="*70 + "\n")
    
    nexus = LycheetahNexus(user_id=user_id)
    
    print("\nCommands:")
    print("  verify <text>  - Verify text")
    print("  load <file>    - Load and verify file")
    print("  report         - Generate session report")
    print("  quit           - Exit session\n")
    
    while True:
        try:
            cmd = input("lycheetah> ").strip()
            
            if not cmd:
                continue
            
            if cmd == "quit" or cmd == "exit":
                print("\nGenerating final report...")
                report = nexus.generate_session_report(save=True)
                nexus.print_report_summary(report)
                print("✓ Session complete")
                break
            
            elif cmd == "report":
                report = nexus.generate_session_report(save=False)
                nexus.print_report_summary(report)
            
            elif cmd.startswith("verify "):
                text = cmd[7:]
                result = nexus.verify_content(text)
                print(f"\nVerdict: {result['verdict']['status']}")
                print(f"Authenticity: {result['verdict']['authenticity_score']}")
            
            elif cmd.startswith("load "):
                filepath = cmd[5:]
                try:
                    with open(filepath, 'r') as f:
                        text = f.read()
                    result = nexus.verify_content(text)
                    print(f"\nVerdict: {result['verdict']['status']}")
                except Exception as e:
                    print(f"Error: {e}")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Generating report...")
            report = nexus.generate_session_report(save=True)
            nexus.print_report_summary(report)
            break
        except Exception as e:
            print(f"Error: {e}")


def show_config():
    """Show current configuration"""
    print("="*70)
    print("LYCHEETAH × CASCADE CONFIGURATION")
    print("="*70 + "\n")
    
    print("SYSTEM:")
    print(f"  Version: {SystemConfig.VERSION}")
    print(f"  Name: {SystemConfig.SYSTEM_NAME}")
    print(f"  Trinity: {', '.join(SystemConfig.TRINITY_AXIOMS)}")
    
    print("\nTHRESHOLDS:")
    thresholds = SignatureThresholds.get_all()
    for key, value in thresholds.items():
        print(f"  {key}: {value}")
    
    print("\nPREDICTED PERFORMANCE:")
    print("  ✓ Authentic Lycheetah content should pass")
    print("  ✓ Generic technical content should fail")
    print("  ⚠️ Requires empirical validation with your actual work")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Lycheetah × CASCADE Signature Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s verify "The Protector ensures safety"
  %(prog)s verify-file document.txt
  %(prog)s session
  %(prog)s config
        """
    )
    
    parser.add_argument('command', 
                       choices=['verify', 'verify-file', 'session', 'config'],
                       help='Command to execute')
    parser.add_argument('input', nargs='?', help='Text or file to verify')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('-u', '--user', default='cli_user',
                       help='User ID for session')
    
    args = parser.parse_args()
    
    if args.command == 'verify':
        if not args.input:
            print("Error: Please provide text to verify")
            sys.exit(1)
        verify_text(args.input, args.verbose)
    
    elif args.command == 'verify-file':
        if not args.input:
            print("Error: Please provide file path")
            sys.exit(1)
        verify_file(args.input, args.verbose)
    
    elif args.command == 'session':
        start_session(args.user)
    
    elif args.command == 'config':
        show_config()


if __name__ == "__main__":
    main()
