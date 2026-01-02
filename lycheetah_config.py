"""
lycheetah_config.py
===================
Configuration file for Lycheetah × CASCADE system

CRITICAL: These thresholds have been tuned based on empirical testing.
The original thresholds were too strict (even authentic content failed).

Author: Lycheetah × CASCADE
Version: 2.0.1
"""

# ============================================================================
# SIGNATURE VERIFICATION THRESHOLDS
# ============================================================================

class SignatureThresholds:
    """
    Thresholds for signature verification
    
    TUNED VALUES (v2.0.1):
    - Based on testing authentic Lycheetah content
    - Original thresholds caused 100% false negatives
    - New thresholds balance detection vs false positives
    """
    
    # Lore Coherence Score (original Lycheetah metric)
    LCS_THRESHOLD = 0.6  # Down from 0.7
    
    # Truth Pressure (Π = TES × VTR)
    TRUTH_PRESSURE_THRESHOLD = 0.45  # Down from 1.2 (CRITICAL FIX)
    
    # Philosophical/Aesthetic Impact
    PAI_THRESHOLD = 0.35  # Down from 0.6
    
    # Composite Authenticity Score
    AUTHENTICITY_THRESHOLD = 0.5  # Down from 0.7
    
    @classmethod
    def get_all(cls) -> dict:
        """Return all thresholds as dict"""
        return {
            'lcs': cls.LCS_THRESHOLD,
            'truth_pressure': cls.TRUTH_PRESSURE_THRESHOLD,
            'pai': cls.PAI_THRESHOLD,
            'authenticity': cls.AUTHENTICITY_THRESHOLD
        }
    
    @classmethod
    def get_strict(cls) -> dict:
        """Stricter thresholds for high-confidence verification"""
        return {
            'lcs': 0.7,
            'truth_pressure': 0.6,
            'pai': 0.5,
            'authenticity': 0.65
        }
    
    @classmethod
    def get_permissive(cls) -> dict:
        """More permissive for maximum detection"""
        return {
            'lcs': 0.5,
            'truth_pressure': 0.3,
            'pai': 0.25,
            'authenticity': 0.4
        }


# ============================================================================
# AXIOM SEMANTIC FIELDS (EXPANDED)
# ============================================================================

class AxiomSemantics:
    """
    Expanded semantic word lists for Trinity detection
    
    UPGRADE v2.0.1:
    - Expanded from 6 words to 25-30 per axiom
    - Covers more expressions of each concept
    - Improves detection without keyword stuffing
    """
    
    PROTECTOR_WORDS = [
        # Core terms
        'protector', 'protect', 'protection',
        # Defense
        'defense', 'defend', 'defensive', 'guardian', 'guard',
        # Safety
        'safety', 'safe', 'safeguard', 'secure', 'security',
        # Sacrifice
        'sacrifice', 'unconditional', 'anchor', 'anchoring',
        # Military/fortress
        'fortress', 'bulwark', 'shield', 'armor', 'barrier',
        # Action
        'ensures', 'maintains', 'preserves', 'upholds'
    ]
    
    HEALER_WORDS = [
        # Core terms
        'healer', 'heal', 'healing',
        # Transformation
        'transmute', 'transform', 'transformation', 'metamorphosis',
        # Alchemy
        'alchemy', 'alchemical', 'alchemist',
        # Structure
        'structure', 'structured', 'restructure',
        # Entropy
        'entropy', 'chaos', 'disorder',
        # Recovery
        'recovery', 'restore', 'restoration', 'repair', 'mend',
        # Renewal
        'renewal', 'regenerate', 'rejuvenate', 'revive',
        # Process
        'process', 'transmutes', 'converts', 'refines'
    ]
    
    BEACON_WORDS = [
        # Core terms
        'beacon', 'light', 'illuminate',
        # Clarity
        'clarity', 'clear', 'transparent',
        # Truth
        'truth', 'veritas', 'true',
        # Eternal
        'eternal', 'everlasting', 'enduring', 'perpetual',
        # Core
        'core', 'central', 'essence', 'fundamental',
        # Signal
        'signal', 'guide', 'guidance', 'waypoint',
        # Navigation
        'lighthouse', 'lodestar', 'compass', 'north-star',
        # Action
        'maintains', 'radiates', 'shines', 'illuminates'
    ]
    
    BRAND_VOICE_WORDS = {
        # Categories of Lycheetah voice intensity
        'strength': ['forge', 'forged', 'storm-forged', 'immutable', 'sovereign'],
        'transformation': ['catalyst', 'paradigm', 'shift', 'evolution'],
        'clarity': ['precision', 'exact', 'mathematical', 'deterministic'],
        'architecture': ['nexus', 'kernel', 'architecture', 'framework', 'protocol'],
        'philosophy': ['axiom', 'principle', 'foundation', 'pillar', 'trinity']
    }
    
    @classmethod
    def get_all_protector(cls) -> list:
        return cls.PROTECTOR_WORDS
    
    @classmethod
    def get_all_healer(cls) -> list:
        return cls.HEALER_WORDS
    
    @classmethod
    def get_all_beacon(cls) -> list:
        return cls.BEACON_WORDS
    
    @classmethod
    def get_brand_voice_all(cls) -> list:
        """Get all brand voice words flattened"""
        all_words = []
        for category_words in cls.BRAND_VOICE_WORDS.values():
            all_words.extend(category_words)
        return all_words


# ============================================================================
# SCORING WEIGHTS
# ============================================================================

class ScoringWeights:
    """
    Weights for various scoring components
    
    ADJUSTED v2.0.1:
    - Reduced symmetry weight (not all content needs it)
    - Increased axiom presence weight
    - Better reflects real-world usage
    """
    
    # LCS Calculation
    LCS_AXIOM_WEIGHT = 0.6  # Up from 0.4
    LCS_SYMMETRY_WEIGHT = 0.4  # Down from 0.6
    
    # TES Components
    TES_DENSITY_WEIGHT = 0.5
    TES_SEMANTIC_WEIGHT = 0.3
    TES_DISTRIBUTION_WEIGHT = 0.2
    
    # VTR Components
    VTR_SYMMETRY_WEIGHT = 0.5
    VTR_CONSISTENCY_WEIGHT = 0.3
    VTR_CLAUSE_WEIGHT = 0.2
    
    # PAI Components
    PAI_POWER_DENSITY_WEIGHT = 0.5
    PAI_TRINITY_BONUS = 0.3
    PAI_INTENSITY_WEIGHT = 0.2
    
    # Authenticity Score
    AUTH_LCS_WEIGHT = 0.4
    AUTH_TRUTH_PRESSURE_WEIGHT = 0.4
    AUTH_PAI_WEIGHT = 0.2


# ============================================================================
# RESONANCE THRESHOLDS
# ============================================================================

class ResonanceThresholds:
    """Thresholds for collaboration quality monitoring"""
    
    # Trinity balance (lower std = more balanced)
    BALANCED_TRINITY_THRESHOLD = 0.7
    IMBALANCED_TRINITY_THRESHOLD = 0.4
    
    # Brand coherence
    STRONG_COHERENCE = 0.6
    WEAK_COHERENCE = 0.3
    
    # User sovereignty
    HEALTHY_SOVEREIGNTY = 0.7
    AT_RISK_SOVEREIGNTY = 0.5
    DEPENDENT_THRESHOLD = 0.3
    
    # Codependency risk
    LOW_DEPENDENCY = 0.3
    MODERATE_DEPENDENCY = 0.5
    HIGH_DEPENDENCY = 0.7


# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

class SystemConfig:
    """General system configuration"""
    
    # Version
    VERSION = "2.0.1"
    SYSTEM_NAME = "Lycheetah × CASCADE"
    
    # Axioms
    TRINITY_AXIOMS = ["Protector", "Healer", "Beacon"]
    
    # LAMAGUE symbols
    LAMAGUE_MAP = {
        "Protector": "Ao",
        "Healer": "Ψ",
        "Beacon": "Ω_heal"
    }
    
    # File paths
    DEFAULT_OUTPUT_DIR = "./lycheetah_data"
    
    # State stabilization (Schmitt Trigger)
    SCHMITT_HIGH_THRESHOLD = 0.85
    SCHMITT_LOW_THRESHOLD = 0.4
    
    # Meta-learning
    ENABLE_META_LEARNING = True
    MIN_SAMPLES_FOR_LEARNING = 50
    
    # Reporting
    VERBOSE_REPORTS = False
    SAVE_HISTORY = True


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LYCHEETAH × CASCADE CONFIGURATION")
    print("="*70 + "\n")
    
    print("SIGNATURE THRESHOLDS:")
    for key, value in SignatureThresholds.get_all().items():
        print(f"  {key}: {value}")
    
    print("\nAXIOM WORD COUNTS:")
    print(f"  Protector: {len(AxiomSemantics.PROTECTOR_WORDS)} words")
    print(f"  Healer: {len(AxiomSemantics.HEALER_WORDS)} words")
    print(f"  Beacon: {len(AxiomSemantics.BEACON_WORDS)} words")
    print(f"  Brand Voice: {len(AxiomSemantics.get_brand_voice_all())} words")
    
    print("\nSCORING WEIGHTS:")
    print(f"  LCS: {ScoringWeights.LCS_AXIOM_WEIGHT} axiom + {ScoringWeights.LCS_SYMMETRY_WEIGHT} symmetry")
    print(f"  Authenticity: {ScoringWeights.AUTH_LCS_WEIGHT} LCS + {ScoringWeights.AUTH_TRUTH_PRESSURE_WEIGHT} Π + {ScoringWeights.AUTH_PAI_WEIGHT} PAI")
    
    print("\nSYSTEM INFO:")
    print(f"  Version: {SystemConfig.VERSION}")
    print(f"  Trinity: {', '.join(SystemConfig.TRINITY_AXIOMS)}")
    
    print("\n" + "="*70)
    print("✓ Configuration loaded successfully")
    print("="*70)
