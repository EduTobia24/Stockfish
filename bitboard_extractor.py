#!/usr/bin/env python3
"""
Chess Position to Bitboard Feature Extractor
============================================

Converts FEN positions to bitboard representation for symbolic regression.
Pure bitboard approach: 12 piece types × 64 squares + game state = 782 features
"""

import numpy as np
import json
import csv
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class ChessBitboardExtractor:
    """
    Converts chess positions (FEN) to bitboard feature vectors.
    
    Feature Layout:
    - 768 piece features: 12 piece types × 64 squares (binary)
    - 14 game state features: castling, en passant, turn
    - Total: 782 features per position
    """
    
    def __init__(self):
        # Piece mapping for FEN parsing
        self.piece_to_index = {
            'P': 0,  # White Pawn
            'R': 1,  # White Rook  
            'N': 2,  # White Knight
            'B': 3,  # White Bishop
            'Q': 4,  # White Queen
            'K': 5,  # White King
            'p': 6,  # Black Pawn
            'r': 7,  # Black Rook
            'n': 8,  # Black Knight
            'b': 9,  # Black Bishop
            'q': 10, # Black Queen
            'k': 11  # Black King
        }
        
        # Square mapping (a1=0, b1=1, ..., h8=63)
        self.files = 'abcdefgh'
        self.ranks = '12345678'
        
    def square_to_index(self, square: str) -> int:
        """Convert algebraic notation (e.g., 'e4') to square index (0-63)"""
        if len(square) != 2:
            return -1
        file_idx = self.files.index(square[0].lower())
        rank_idx = self.ranks.index(square[1])
        return rank_idx * 8 + file_idx
    
    def index_to_square(self, index: int) -> str:
        """Convert square index (0-63) to algebraic notation"""
        rank = index // 8
        file = index % 8
        return self.files[file] + self.ranks[rank]
    
    def fen_to_bitboards(self, fen: str) -> np.ndarray:
        """
        Convert FEN string to bitboard feature vector.
        
        Args:
            fen: FEN string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            
        Returns:
            numpy array of shape (782,) with binary features
        """
        parts = fen.split(' ')
        if len(parts) != 6:
            raise ValueError(f"Invalid FEN format: {fen}")
        
        board_fen, turn, castling, en_passant, halfmove, fullmove = parts
        
        # Initialize feature vector: 12 piece bitboards × 64 squares + 14 game state
        features = np.zeros(782, dtype=np.float32)
        
        # Parse piece positions (768 features: 0-767)
        self._parse_piece_positions(board_fen, features)
        
        # Parse game state (14 features: 768-781)
        self._parse_game_state(turn, castling, en_passant, features)
        
        return features
    
    def _parse_piece_positions(self, board_fen: str, features: np.ndarray) -> None:
        """Parse piece positions from board FEN into bitboard features."""
        square_idx = 56  # Start at a8 (top-left in FEN notation)
        
        for char in board_fen:
            if char == '/':
                square_idx -= 16  # Move to next rank (down 8, then back 8)
                continue
            elif char.isdigit():
                square_idx += int(char)  # Skip empty squares
                continue
            elif char in self.piece_to_index:
                # Place piece on square
                piece_type = self.piece_to_index[char]
                feature_idx = piece_type * 64 + square_idx
                features[feature_idx] = 1.0
                square_idx += 1
            else:
                raise ValueError(f"Invalid FEN character: {char}")
    
    def _parse_game_state(self, turn: str, castling: str, en_passant: str, features: np.ndarray) -> None:
        """Parse game state into features 768-781."""
        base_idx = 768
        
        # Turn to move (1 feature: index 768)
        features[base_idx] = 1.0 if turn == 'w' else 0.0
        
        # Castling rights (4 features: indices 769-772)
        features[base_idx + 1] = 1.0 if 'K' in castling else 0.0  # White kingside
        features[base_idx + 2] = 1.0 if 'Q' in castling else 0.0  # White queenside
        features[base_idx + 3] = 1.0 if 'k' in castling else 0.0  # Black kingside
        features[base_idx + 4] = 1.0 if 'q' in castling else 0.0  # Black queenside
        
        # En passant target (8 features: indices 773-780, one-hot encoded by file)
        if en_passant != '-' and len(en_passant) == 2:
            file_idx = self.files.index(en_passant[0].lower())
            features[base_idx + 5 + file_idx] = 1.0
        # If no en passant, all 8 features remain 0
        
        # Feature 781 reserved for future use (currently 0)
    
    def extract_dataset_features(self, dataset_path: str, filter_extreme=False, eval_threshold=20.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract bitboard features from entire dataset.
        
        Args:
            dataset_path: Path to JSON dataset file
            filter_extreme: If True, filter out evaluations beyond ±eval_threshold
            eval_threshold: Evaluation threshold for filtering (default: 20.0)
            
        Returns:
            Tuple of (features, evaluations)
            - features: shape (N, 782) - bitboard features
            - evaluations: shape (N,) - target evaluations
        """
        print(f"🔄 Loading dataset from {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        n_positions = len(dataset)
        print(f"📊 Processing {n_positions} positions...")
        
        if filter_extreme:
            print(f"🔍 Filtering evaluations beyond ±{eval_threshold}")
        
        # Initialize arrays (with buffer for filtering)
        features = np.zeros((n_positions, 782), dtype=np.float32)
        evaluations = np.zeros(n_positions, dtype=np.float32)
        
        successful_extractions = 0
        filtered_count = 0
        
        for i, position in enumerate(dataset):
            try:
                # Extract features from FEN
                fen = position['fen']
                features[i] = self.fen_to_bitboards(fen)
                
                # Extract target evaluation
                eval_score = position['evaluation']
                
                # Handle mate scores (convert to large finite values)
                if position.get('mate_in') is not None:
                    mate_in = position['mate_in']
                    if mate_in > 0:
                        eval_score = 20.0 - 0.1 * mate_in  # White mate: ~20
                    else:
                        eval_score = -20.0 - 0.1 * mate_in  # Black mate: ~-20
                
                # Apply filtering if enabled
                if filter_extreme and (eval_score > eval_threshold or eval_score < -eval_threshold):
                    filtered_count += 1
                    continue  # Skip this position
                
                evaluations[successful_extractions] = eval_score
                features[successful_extractions] = self.fen_to_bitboards(fen)
                successful_extractions += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{n_positions} positions...")
                    
            except Exception as e:
                print(f"  ⚠️  Error processing position {i+1}: {e}")
                # Skip failed positions
                
        # Trim arrays to actual size
        features = features[:successful_extractions]
        evaluations = evaluations[:successful_extractions]
        
        print(f"✅ Extracted features from {successful_extractions}/{n_positions} positions")
        if filter_extreme:
            print(f"🔍 Filtered out {filtered_count} positions with extreme evaluations (±{eval_threshold})")
        print(f"📊 Final dataset: {len(features)} positions")
        
        return features, evaluations
    
    def save_features_csv(self, features: np.ndarray, evaluations: np.ndarray, 
                         output_path: str) -> None:
        """Save extracted features to CSV file for analysis."""
        print(f"💾 Saving features to {output_path}")
        
        # Create column names
        column_names = []
        
        # Piece bitboard columns (768 features)
        piece_names = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
        for piece_idx, piece in enumerate(piece_names):
            for square_idx in range(64):
                square = self.index_to_square(square_idx)
                column_names.append(f"{piece}_{square}")
        
        # Game state columns (14 features)
        column_names.extend([
            'white_to_move',
            'castle_K', 'castle_Q', 'castle_k', 'castle_q',
            'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h',
            'reserved'
        ])
        
        # Add evaluation column
        column_names.append('evaluation')
        
        # Combine features and evaluations
        data = np.column_stack([features, evaluations])
        
        # Write to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            
            for row in data:
                writer.writerow(row)
        
        print(f"✅ Saved {len(data)} rows × {len(column_names)} columns")
    
    def analyze_feature_sparsity(self, features: np.ndarray) -> None:
        """Analyze sparsity of bitboard features."""
        print("\n" + "=" * 60)
        print("📊 BITBOARD FEATURE ANALYSIS")
        print("=" * 60)
        
        n_positions, n_features = features.shape
        
        # Overall sparsity
        total_nonzero = np.count_nonzero(features)
        total_elements = n_positions * n_features
        sparsity = 1.0 - (total_nonzero / total_elements)
        
        print(f"📍 Total positions: {n_positions}")
        print(f"🔢 Total features: {n_features}")
        print(f"✨ Sparsity: {sparsity:.1%} (most features are 0)")
        
        # Piece-specific analysis
        piece_names = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
        
        print(f"\n🔍 Piece Occupancy Analysis:")
        for i, piece in enumerate(piece_names):
            piece_features = features[:, i*64:(i+1)*64]
            piece_count = np.count_nonzero(piece_features)
            avg_per_position = piece_count / n_positions
            print(f"   {piece}: {avg_per_position:.1f} avg per position")
        
        # Game state analysis
        game_state_features = features[:, 768:]
        print(f"\n🎮 Game State Features:")
        print(f"   White to move: {np.mean(game_state_features[:, 0]):.1%}")
        print(f"   Castling rights: {np.mean(game_state_features[:, 1:5]):.1%} avg")
        print(f"   En passant: {np.mean(game_state_features[:, 5:13]):.1%} avg")
        
        print("=" * 60)
    
    def get_feature_names(self) -> List[str]:
        """Get human-readable feature names for debugging/analysis."""
        names = []
        
        # Piece bitboard names
        piece_names = ['WP', 'WR', 'WN', 'WB', 'WQ', 'WK', 'BP', 'BR', 'BN', 'BB', 'BQ', 'BK']
        for piece in piece_names:
            for square_idx in range(64):
                square = self.index_to_square(square_idx)
                names.append(f"{piece}_{square}")
        
        # Game state names
        names.extend([
            'white_to_move',
            'castle_K', 'castle_Q', 'castle_k', 'castle_q',
            'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h',
            'reserved'
        ])
        
        return names

def main():
    """Extract bitboard features from dataset and prepare for SR training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract bitboard features from chess dataset')
    parser.add_argument('--dataset', default='stockfish_dataset.json', 
                       help='Path to JSON dataset file')
    parser.add_argument('--output', default='bitboard_features',
                       help='Output filename prefix')
    parser.add_argument('--test-fen', 
                       help='Test single FEN position (for debugging)')
    parser.add_argument('--filter-extreme', action='store_true',
                       help='Filter out evaluations beyond ±threshold (default: ±20)')
    parser.add_argument('--eval-threshold', type=float, default=20.0,
                       help='Evaluation threshold for filtering (default: 20.0)')
    
    args = parser.parse_args()
    
    extractor = ChessBitboardExtractor()
    
    # Test mode: process single FEN
    if args.test_fen:
        print(f"🧪 Testing FEN: {args.test_fen}")
        try:
            features = extractor.fen_to_bitboards(args.test_fen)
            print(f"✅ Extracted {len(features)} features")
            print(f"   Non-zero features: {np.count_nonzero(features)}")
            print(f"   Sparsity: {1.0 - np.count_nonzero(features)/len(features):.1%}")
            
            # Show non-zero features
            feature_names = extractor.get_feature_names()
            nonzero_indices = np.nonzero(features)[0]
            print(f"\n🔍 Active features:")
            for idx in nonzero_indices[:20]:  # Show first 20
                print(f"   {feature_names[idx]}: {features[idx]}")
            if len(nonzero_indices) > 20:
                print(f"   ... and {len(nonzero_indices) - 20} more")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        return
    
    # Dataset mode: process full dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("🚀 Chess Bitboard Feature Extractor")
    print("=" * 60)
    print(f"📂 Dataset: {dataset_path}")
    print(f"📊 Output: {args.output}")
    if args.filter_extreme:
        print(f"🔍 Filtering: Remove evaluations beyond ±{args.eval_threshold}")
    print("=" * 60)
    
    # Extract features
    features, evaluations = extractor.extract_dataset_features(
        str(dataset_path), 
        filter_extreme=args.filter_extreme,
        eval_threshold=args.eval_threshold
    )
    
    if len(features) == 0:
        print("❌ No features extracted!")
        return
    
    # Analyze features
    extractor.analyze_feature_sparsity(features)
    
    # Save to multiple formats
    print(f"\n💾 Saving extracted features...")
    
    # Save as NPZ (efficient binary format)
    npz_path = f"{args.output}.npz"
    np.savez_compressed(npz_path, features=features, evaluations=evaluations)
    print(f"   📦 NPZ (binary): {npz_path}")
    
    # Save as CSV (human-readable)
    csv_path = f"{args.output}.csv"
    extractor.save_features_csv(features, evaluations, csv_path)
    print(f"   📊 CSV (readable): {csv_path}")
    
    # Save feature info
    info_path = f"{args.output}_info.json"
    info = {
        'n_positions': len(features),
        'n_features': len(features[0]),
        'feature_layout': {
            'piece_bitboards': '0-767 (12 pieces × 64 squares)',
            'game_state': '768-781 (turn, castling, en passant)',
            'total_features': 782
        },
        'sparsity': float(1.0 - np.count_nonzero(features) / features.size),
        'evaluation_stats': {
            'min': float(np.min(evaluations)),
            'max': float(np.max(evaluations)),
            'mean': float(np.mean(evaluations)),
            'std': float(np.std(evaluations))
        }
    }
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"   ℹ️  Info: {info_path}")
    
    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Ready for symbolic regression training:")
    print(f"   - Features shape: {features.shape}")
    print(f"   - Evaluations shape: {evaluations.shape}")
    print(f"   - Sparsity: {1.0 - np.count_nonzero(features) / features.size:.1%}")
    print(f"\n🎯 Next step: Train PySR model with these features!")

if __name__ == "__main__":
    main()