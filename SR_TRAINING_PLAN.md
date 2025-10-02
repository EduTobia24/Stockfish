# Symbolic Regression Training Plan
## Chess Position Evaluation with Auto-Improving Engine

### Project Overview
Transform Stockfish's NNUE evaluation into a symbolic regression model that can:
- **Learn** meaningful chess patterns from position data
- **Improve** automatically through training on new positions  
- **Generalize** to unseen positions with interpretable formulas
- **Evolve** evaluation functions that are human-readable

---

## Input Representation Strategies

### 1. **Bitboard Representation** ⭐ (Recommended)
**Concept**: Represent each piece type as binary features
```python
# 12 bitboards (6 pieces × 2 colors) × 64 squares = 768 binary features
features = [
    white_pawns[64],    # [0,1,0,1,0,0,...] 64 squares
    white_rooks[64],    
    white_knights[64],
    white_bishops[64],
    white_queens[64],
    white_kings[64],
    black_pawns[64],
    black_rooks[64],
    black_knights[64], 
    black_bishops[64],
    black_queens[64],
    black_kings[64],
    # Plus game state features
    white_to_move,      # 1 binary
    castle_kingside_w,  # 4 binary castling rights
    castle_queenside_w,
    castle_kingside_b,
    castle_queenside_b,
    en_passant_file     # 8 binary (which file, if any)
]
# Total: 768 + 14 = 782 features
```

**Advantages**:
- ✅ **Native chess representation** (same as engines use)
- ✅ **Sparse and efficient** (most squares empty)
- ✅ **Directly interpretable** (each bit = piece on square)
- ✅ **Captures spatial relationships** naturally

**Disadvantages**:
- ❌ **High dimensionality** (782 features)
- ❌ **Sparse data** (most features are 0)

---

### 2. **Piece-Square Tables** ⭐⭐ (Highly Recommended)
**Concept**: Classical chess evaluation features
```python
features = [
    # Material count (6 features)
    white_pawns_count, white_rooks_count, white_knights_count,
    white_bishops_count, white_queens_count, white_kings_count,
    black_pawns_count, black_rooks_count, black_knights_count, 
    black_bishops_count, black_queens_count, black_kings_count,
    
    # Piece-square values (384 features: 6 pieces × 2 colors × 32 unique squares)
    white_pawn_square_values[32],   # Exploit symmetry (a1=h1, a2=h2, etc)
    white_rook_square_values[32],
    # ... for each piece type
    
    # Positional features (20-50 features)
    center_control,           # Pieces attacking central squares
    king_safety_white,        # Distance to enemy pieces
    king_safety_black,
    pawn_structure_score,     # Doubled, isolated, passed pawns
    piece_mobility,           # Number of legal moves
    bishop_pair_bonus,        # Having both bishops
    rook_on_open_file,        # Rooks on files without pawns
    
    # Game phase (2 features)
    opening_phase_weight,     # Based on material remaining
    endgame_phase_weight,
    
    # Tactical features (10-20 features)
    pins_and_skewers,
    discovered_attacks,
    fork_opportunities,
    
    # Meta features (5 features)
    white_to_move,
    castling_rights_score,
    en_passant_target
]
# Total: ~450-500 carefully chosen features
```

**Advantages**:
- ✅ **Chess domain knowledge** incorporated
- ✅ **Manageable dimensionality** (500 vs 782)
- ✅ **Meaningful features** for SR to combine
- ✅ **Proven effective** in traditional engines

---

### 3. **Hybrid Neural-Symbolic Features** ⭐⭐⭐ (Best Performance)
**Concept**: Extract features from Stockfish's NNUE network
```python
# Use Stockfish's internal NNUE feature extraction
features = [
    # Raw NNUE features (reduce from 22,528 to ~100-200)
    nnue_king_features[50],      # King safety patterns
    nnue_pawn_features[30],      # Pawn structure patterns  
    nnue_piece_features[40],     # Piece coordination patterns
    nnue_tactical_features[20],  # Tactical motifs
    
    # Classical features (as above)
    material_features[12],
    positional_features[40],
    tactical_features[20],
    
    # Derived features
    phase_transition_score,
    king_pawn_storm_score,
    weak_square_control
]
# Total: ~200-250 highly informative features
```

**Advantages**:
- ✅ **Best of both worlds** (neural + symbolic)
- ✅ **Pre-learned patterns** from NNUE
- ✅ **Compact representation** 
- ✅ **High-quality features** for SR

---

### 4. **Compressed Position Encoding**
**Concept**: Use dimensionality reduction on raw positions
```python
# Apply PCA/autoencoders to reduce 782 → ~50-100 features
features = compressed_position_vector[100]  # Principal components
```

**Advantages**:
- ✅ **Compact representation**
- ✅ **Captures main variations**

**Disadvantages**:
- ❌ **Loses interpretability** (black box features)
- ❌ **May lose chess-specific patterns**

---

## Recommended Approach: Multi-Stage Training

### Stage 1: Classical Features (Baseline)
```python
# Start with interpretable piece-square features (~500 features)
features_v1 = extract_classical_features(fen)
sr_model_v1 = train_pysr(features_v1, evaluations)
```

### Stage 2: Hybrid Enhancement  
```python
# Add NNUE-derived features (~200 total features)
features_v2 = combine(
    extract_classical_features(fen),
    extract_nnue_features(fen, stockfish_nnue)
)
sr_model_v2 = train_pysr(features_v2, evaluations)
```

### Stage 3: Auto-Discovery
```python
# Let SR discover new feature combinations
features_v3 = features_v2 + discover_new_features(sr_model_v2)
sr_model_v3 = train_pysr(features_v3, evaluations)
```

---

## Training Pipeline Architecture

### Data Flow
```
Chess Position (FEN) 
    ↓
Feature Extraction Engine
    ↓
Feature Vector [200-500 dims]
    ↓
PySR Symbolic Regression
    ↓
Symbolic Formula: f(features) = evaluation
    ↓
Integration back into Stockfish
```

### Implementation Plan

#### Phase 1: Feature Engineering (Week 1-2)
1. **Build feature extractor** from FEN strings
2. **Implement classical chess features**
3. **Extract NNUE features** from Stockfish
4. **Validate feature quality** against known positions

#### Phase 2: Initial SR Training (Week 3)
1. **Train PySR model** on 1000-position dataset
2. **Evaluate accuracy** vs Stockfish evaluations
3. **Analyze discovered formulas** for chess insights
4. **Optimize hyperparameters**

#### Phase 3: Integration (Week 4)
1. **Replace NNUE evaluation** with SR formula
2. **Test performance** in actual games
3. **Benchmark against original** Stockfish
4. **Measure ELO difference**

#### Phase 4: Auto-Improvement (Week 5+)
1. **Generate new positions** from self-play
2. **Retrain SR model** periodically
3. **Version control** for evaluation functions
4. **Monitor performance evolution**

---

## Technical Implementation

### Feature Extraction Framework
```python
class ChessFeatureExtractor:
    def __init__(self, use_nnue=True):
        self.use_nnue = use_nnue
        self.stockfish = Stockfish() if use_nnue else None
    
    def extract_features(self, fen: str) -> np.ndarray:
        """Extract feature vector from chess position"""
        features = []
        
        # Classical features
        features.extend(self.extract_material(fen))
        features.extend(self.extract_positional(fen))
        features.extend(self.extract_tactical(fen))
        
        # NNUE features (if enabled)
        if self.use_nnue:
            features.extend(self.extract_nnue_features(fen))
        
        return np.array(features)
    
    def extract_material(self, fen: str) -> List[float]:
        """Material balance and piece counts"""
        # Implementation here
        pass
    
    def extract_positional(self, fen: str) -> List[float]:
        """King safety, pawn structure, piece activity"""
        # Implementation here  
        pass
    
    def extract_nnue_features(self, fen: str) -> List[float]:
        """Extract relevant NNUE network activations"""
        # Implementation here
        pass
```

### PySR Training Configuration
```python
from pysr import PySRRegressor

sr_model = PySRRegressor(
    niterations=1000,           # Training iterations
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "abs", "tanh"],
    model_selection="best",     # Choose best accuracy
    loss="L2DistLoss()",       # Mean squared error
    populations=50,             # Parallel evolution populations  
    population_size=100,        # Individuals per population
    max_complexity=20,          # Formula complexity limit
    parsimony=0.0032,          # Simplicity preference
    feature_selection=True,     # Auto-select relevant features
    procs=4,                   # Parallel processes
    multithreading=True,
    random_state=42
)
```

---

## Success Metrics

### Accuracy Metrics
- **MSE vs Stockfish**: < 0.5 pawns difference
- **Correlation**: R² > 0.85 with Stockfish evaluations  
- **Mate Detection**: 95%+ accuracy on tactical positions

### Performance Metrics  
- **ELO Rating**: Within 50-100 ELO of original Stockfish
- **Speed**: Evaluation time < 10× slower than NNUE
- **Memory**: Formula size < 1MB vs 133MB NNUE

### Interpretability Metrics
- **Formula Complexity**: < 20 terms in final expression
- **Chess Validity**: Discovered patterns match known theory
- **Human Readability**: Grandmasters can understand formula

---

## Next Steps

1. **Choose initial approach**: Start with **Piece-Square Tables** (Option 2)
2. **Build feature extractor** for the 1000-position dataset
3. **Train first SR model** and analyze results
4. **Iteratively improve** feature engineering
5. **Scale to larger datasets** as we get promising results

Would you like me to start implementing the feature extraction framework for our dataset?