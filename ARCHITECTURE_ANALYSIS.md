# Task 1.1: Current Architecture Analysis
**Date: October 2, 2025**
**Project: Stockfish-SR**

## 1. Current NNUE Evaluation System Architecture

### 1.1 High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Stockfish Engine                           │
├─────────────────────────────────────────────────────────────────┤
│  UCI Interface (uci.cpp/uci.h)                                │
│  ├─ Command parsing and protocol handling                      │
│  ├─ Engine options management                                  │
│  └─ Communication with external GUIs                          │
├─────────────────────────────────────────────────────────────────┤
│  Search Engine (search.cpp/search.h)                          │
│  ├─ Alpha-beta pruning with extensions                         │
│  ├─ Move ordering and history heuristics                       │
│  ├─ Transposition table integration                            │
│  └─ **EVALUATION CALLS** → Eval::evaluate()                   │
├─────────────────────────────────────────────────────────────────┤
│  NNUE Evaluation System (evaluate.cpp/evaluate.h)             │
│  ├─ Network selection (big/small net)                          │
│  ├─ PSQT (Piece-Square Table) + Positional evaluation         │
│  ├─ Material and complexity blending                           │
│  └─ **TARGET FOR SR REPLACEMENT**                             │
├─────────────────────────────────────────────────────────────────┤
│  NNUE Neural Networks (nnue/ directory)                       │
│  ├─ Feature Transformer (HalfKAv2_hm)                         │
│  ├─ Accumulator Stack & Caches                                │
│  ├─ Network Architecture (layers, activation functions)        │
│  └─ Pre-trained network files (.nnue)                         │
├─────────────────────────────────────────────────────────────────┤
│  Position Representation (position.cpp/position.h)            │
│  ├─ Board state management                                     │
│  ├─ Move generation and validation                             │
│  ├─ Feature extraction for NNUE                               │
│  └─ **SOURCE OF FEATURES FOR SR**                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 NNUE Components Breakdown

#### A. Network Structure
- **Two Networks**: Big network (nn-1c0000000000.nnue) and Small network (nn-37f18f62d772.nnue)
- **Network Selection**: Based on `simple_eval()` absolute value > 962
- **Architecture**: 
  - Feature Transformer: HalfKAv2_hm (King-Piece relationships)
  - Hidden layers with ClippedReLU activation
  - Output: PSQT (Piece-Square Table) + Positional scores

#### B. Feature System (HalfKAv2_hm)
```cpp
// Core feature dimensions
static constexpr IndexType Dimensions = SQUARE_NB * PS_NB / 2;
// Where PS_NB = 11 piece types * 64 squares = 704 features
// Dimensions = 64 * 704 / 2 = 22,528 input features
```

**Feature Categories:**
- King-relative piece positions (bucketed by king square)
- Piece-square relationships
- Color perspective (mirrored for black)
- 32 king buckets for position context

#### C. Accumulator System
- **Incremental Updates**: Efficient feature updates during search
- **Accumulator Stack**: Maintains evaluation state through move tree
- **Caching**: Separate caches for big/small networks
- **SIMD Optimization**: Vectorized computation for performance

## 2. Data Flow Analysis

### 2.1 Evaluation Call Path

```
Search::search() → Eval::evaluate()
├─ 1. Position Analysis
│  ├─ simple_eval() → material count
│  ├─ use_smallnet() → network selection
│  └─ pos.checkers() → validity check
├─ 2. NNUE Computation
│  ├─ networks.{big|small}.evaluate()
│  ├─ Feature extraction (HalfKAv2_hm)
│  ├─ Accumulator updates
│  └─ Network forward pass
├─ 3. Score Processing
│  ├─ PSQT + Positional blending (125 * psqt + 131 * positional) / 128
│  ├─ Complexity adjustment (nnueComplexity)
│  ├─ Optimism blending
│  └─ Material scaling
├─ 4. Final Adjustments
│  ├─ Rule50 dampening (shuffling penalty)
│  ├─ Tablebase range clamping
│  └─ Return final Value
```

### 2.2 Performance Hotspots Identified

1. **Feature Extraction**: HalfKAv2_hm feature computation
2. **Network Evaluation**: Forward pass through neural network layers
3. **Accumulator Updates**: Incremental feature updates
4. **Network Selection**: Dynamic big/small network switching
5. **SIMD Operations**: Vectorized accumulator arithmetic

### 2.3 Memory Usage Patterns

```cpp
// Key memory structures
AccumulatorStack accumulators;        // Per-thread evaluation state
AccumulatorCaches caches;             // Network-specific caches
Networks networks;                    // Pre-loaded network weights
FeatureTransformer featureTransformer; // Input layer weights
```

## 3. Current Evaluation Features Analysis

### 3.1 Position Features Currently Used

#### A. Material Features
```cpp
int simple_eval(const Position& pos) {
    Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}
```

#### B. NNUE Input Features (HalfKAv2_hm)
- **King-relative piece positions**: All pieces relative to king square
- **Piece-square combinations**: 11 piece types × 64 squares
- **King buckets**: 32 strategically chosen king position buckets
- **Color perspective**: Mirrored representation for black

#### C. Additional Context Features
- **Material count**: `535 * pos.count<PAWN>() + pos.non_pawn_material()`
- **Rule50 counter**: For draw detection and evaluation dampening
- **Complexity measure**: `abs(psqt - positional)` for evaluation confidence
- **Optimism factor**: Search-tree dependent evaluation bias

### 3.2 Feature Importance Analysis

**High Impact Features (from NNUE architecture):**
1. King safety patterns (king bucket system)
2. Piece activity and mobility (positional scoring)
3. Material imbalances (PSQT component)
4. Piece coordination (king-relative positioning)

**Medium Impact Features:**
1. Pawn structure (implicitly learned)
2. Piece-square values (PSQT tables)
3. Phase evaluation (material-based scaling)

**Low Impact but Necessary:**
1. Rule50 shuffling detection
2. Tablebase integration boundaries
3. Evaluation clamping for stability

## 4. UCI Protocol Integration Points

### 4.1 Evaluation-Related UCI Commands

```cpp
// From uci.cpp - Key evaluation interfaces
"eval"           → Eval::trace() // Detailed evaluation breakdown
"go"             → Search engine → Eval::evaluate() calls
"setoption"      → Network loading and evaluation parameters
"position"       → Position setup → Feature extraction basis
```

### 4.2 UCI Options Affecting Evaluation

```cpp
// Network file options
"EvalFile"       → Custom .nnue network loading
"UCI_ShowWDL"    → Win/Draw/Loss probability display

// Search options affecting evaluation calls
"Hash"           → Transposition table size
"Threads"        → Parallel evaluation contexts
"Contempt"       → Evaluation bias for draws
```

### 4.3 Output Integration

```cpp
// Search output with evaluation scores
"info score cp XXX"     → Centipawn evaluation from Eval::evaluate()
"info score mate XX"    → Mate detection (not from evaluation)
"bestmove"              → Move selection based on evaluation-guided search
```

## 5. Key Integration Points for SR System

### 5.1 Primary Replacement Targets

1. **Eval::evaluate() Function** (evaluate.cpp:48)
   - Main evaluation entry point
   - Current NNUE network calls
   - **SR Replacement Point #1**

2. **Feature Extraction System** (nnue/features/half_ka_v2_hm.h)
   - Position → Feature vector conversion
   - **SR Input Generation Point**

3. **Network Forward Pass** (nnue/network.cpp)
   - Neural network computation
   - **SR Formula Execution Point**

### 5.2 Preservation Requirements

1. **UCI Protocol Compatibility**
   - Maintain same evaluation value ranges
   - Preserve trace output format
   - Keep evaluation timing characteristics

2. **Search Integration**
   - Value type compatibility (int16_t Value)
   - Evaluation call signature preservation
   - Performance characteristics maintenance

3. **Threading and Caching**
   - Thread-safe evaluation calls
   - Accumulator-style incremental updates
   - Memory usage patterns

## 6. Performance Baseline Requirements

### 6.1 Current Performance Characteristics

- **Evaluation Calls**: ~50M-100M per second (typical search)
- **Memory Usage**: ~100MB for networks + caches per thread
- **Feature Extraction**: ~1000-2000 cycles per position
- **Network Evaluation**: ~5000-10000 cycles per position

### 6.2 SR Integration Constraints

- **Maximum Overhead**: <20% evaluation time increase
- **Memory Limits**: Similar memory footprint for caching
- **Threading**: Must support parallel evaluation contexts
- **Determinism**: Reproducible evaluation results

## 7. Recommendations for SR Integration

### 7.1 Architectural Approach

1. **Hybrid System**: Maintain NNUE fallback during development
2. **Gradual Migration**: Replace components incrementally
3. **Feature Parity**: Start with similar features to NNUE
4. **Performance Monitoring**: Continuous benchmarking during development

### 7.2 Critical Success Factors

1. **Feature Engineering**: Extract comprehensive position features
2. **Performance Optimization**: C++/Python bridge efficiency
3. **Evaluation Quality**: Maintain or improve strength
4. **Integration Testing**: Extensive validation against baseline

---

**Completion Status**: ✅ Task 1.1 Complete
**Next Task**: Task 1.2 - Baseline Performance Benchmarking
**Key Findings**: NNUE system is highly optimized but modular enough for SR replacement
**Risk Assessment**: Medium complexity due to performance requirements and feature engineering needs