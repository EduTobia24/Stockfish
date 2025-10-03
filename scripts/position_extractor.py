#!/usr/bin/env python3
"""
Position-based Feature Extractor
================================

Extracts chess position features using (x,y) coordinates for each piece type
plus game state features like castling, en passant, and turn.

Feature format:
- White Pawn positions: up to 8 pieces × 2 coordinates = 16 features
- White Rook positions: up to 2 pieces × 2 coordinates = 4 features  
- White Knight positions: up to 2 pieces × 2 coordinates = 4 features
- White Bishop positions: up to 2 pieces × 2 coordinates = 4 features
- White Queen positions: up to 1 piece × 2 coordinates = 2 features
- White King position: 1 piece × 2 coordinates = 2 features
- Black pieces: same as white = 32 features
- Total piece positions: 64 features
- Game state: 9 features (turn, castling×4, en passant×4)
- Total: 73 features
"""

import numpy as np
import chess
import chess.engine
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import json
from datetime import datetime

def fen_to_position_features(fen: str) -> np.ndarray:
    """
    Convert a FEN string to position-based features.
    
    Returns:
        np.ndarray: Feature vector of length 73
                   - 64 piece position features (x,y coordinates)
                   - 9 game state features
    """
    board = chess.Board(fen)
    features = np.zeros(73, dtype=np.float32)
    
    # Piece type mapping
    piece_types = {
        chess.PAWN: 0,
        chess.ROOK: 1, 
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    # Max pieces per type
    max_pieces = {
        chess.PAWN: 8,
        chess.ROOK: 2,
        chess.KNIGHT: 2, 
        chess.BISHOP: 2,
        chess.QUEEN: 1,
        chess.KING: 1
    }
    
    # Feature index calculation
    def get_piece_start_idx(piece_type: int, is_white: bool) -> int:
        """Get starting index for piece type in feature vector."""
        base_idx = 0
        
        # Add indices for previous piece types
        for pt in range(piece_type):
            base_idx += max_pieces[pt + 1] * 2  # 2 coordinates per piece
        
        # If black pieces, add all white piece indices
        if not is_white:
            base_idx += 32  # 32 features for all white pieces
            
        return base_idx
    
    # Extract piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Convert square to (x, y) coordinates (1-8 range)
            file = chess.square_file(square) + 1  # a-h -> 1-8
            rank = chess.square_rank(square) + 1  # 1-8 -> 1-8
            
            piece_type = piece.piece_type
            is_white = piece.color == chess.WHITE
            
            # Get starting index for this piece type
            start_idx = get_piece_start_idx(piece_type - 1, is_white)
            
            # Find first available slot for this piece type
            slot_found = False
            for slot in range(max_pieces[piece_type]):
                coord_idx = start_idx + slot * 2
                
                # Check if this slot is empty (both x and y are 0)
                if features[coord_idx] == 0 and features[coord_idx + 1] == 0:
                    features[coord_idx] = file      # x coordinate
                    features[coord_idx + 1] = rank  # y coordinate
                    slot_found = True
                    break
            
            if not slot_found:
                print(f"⚠️ Warning: Too many {piece.symbol()} pieces, skipping one at {chess.square_name(square)}")
    
    # Game state features (starting at index 64)
    game_state_idx = 64
    
    # Turn to move (0 = black, 1 = white)
    features[game_state_idx] = 1.0 if board.turn == chess.WHITE else 0.0
    game_state_idx += 1
    
    # Castling rights (4 features)
    features[game_state_idx] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[game_state_idx + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[game_state_idx + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[game_state_idx + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    game_state_idx += 4
    
    # En passant target (4 features: file a-h as one-hot encoding)
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)  # 0-7 for a-h
        if 0 <= ep_file <= 3:  # Files a-d
            features[game_state_idx + ep_file] = 1.0
    
    return features

def extract_dataset_features_from_json(json_file: str, output_file: str, 
                                     filter_extreme: bool = False, eval_threshold: float = 20.0) -> None:
    """
    Extract position features from JSON evaluation file.
    
    Args:
        json_file: Path to JSON file with evaluations (evaluations_500.json format)
        output_file: Output NPZ file path
        filter_extreme: Whether to filter out extreme evaluations
        eval_threshold: Threshold for extreme evaluation filtering (±pawns)
    """
    print(f"🔍 Extracting position features from {json_file}")
    if filter_extreme:
        print(f"🚫 Filtering evaluations > ±{eval_threshold}")
    
    # Read JSON data
    print(f"📋 Loading JSON data...")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        return
    
    print(f"✅ Loaded {len(data)} positions from JSON")
    
    features_list = []
    evaluations_list = []
    skipped_count = 0
    error_count = 0
    
    for i, entry in enumerate(data):
        try:
            # Extract FEN and evaluation from JSON entry
            fen = entry['fen']
            evaluation = entry['evaluation']
            
            # Apply extreme evaluation filter if enabled
            if filter_extreme and abs(evaluation) > eval_threshold:
                skipped_count += 1
                if i % 1000 == 0:
                    print(f"🚫 Skipped position {i+1}: evaluation {evaluation:+.2f} exceeds ±{eval_threshold}")
                continue
            
            # Extract position features
            position_features = fen_to_position_features(fen)
            
            features_list.append(position_features)
            evaluations_list.append(evaluation)
            
            # Progress update
            if (i + 1) % 1000 == 0:
                processed = len(features_list)
                print(f"✅ Processed {i+1}/{len(data)} positions, kept {processed} (skipped {skipped_count}, errors {error_count})")
                print(f"   Latest: eval={evaluation:+.2f}, features_shape={position_features.shape}")
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Show first few errors
                print(f"❌ Error processing position {i+1}: {e}")
                if error_count <= 3:  # Show FEN for first few errors
                    print(f"   FEN: {entry.get('fen', 'N/A')}")
            elif error_count == 6:
                print(f"❌ ... (suppressing further error messages)")
            continue
    
    if not features_list:
        print(f"❌ No valid positions extracted!")
        return
    
    # Convert to numpy arrays
    features_array = np.array(features_list, dtype=np.float32)
    evaluations_array = np.array(evaluations_list, dtype=np.float32)
    
    print(f"\n📊 Feature Extraction Summary:")
    print(f"   Total positions in JSON: {len(data)}")
    print(f"   Valid positions extracted: {len(features_list)}")
    print(f"   Positions skipped (extreme): {skipped_count}")
    print(f"   Positions with errors: {error_count}")
    print(f"   Success rate: {len(features_list)/len(data)*100:.1f}%")
    print(f"   Feature vector size: {features_array.shape[1]}")
    print(f"   Evaluation range: {evaluations_array.min():+.2f} to {evaluations_array.max():+.2f}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to NPZ file
    np.savez_compressed(output_file, 
                       features=features_array, 
                       evaluations=evaluations_array)
    
    print(f"💾 Saved {len(features_list)} positions to {output_file}")
    print(f"📏 Dataset size: {features_array.nbytes + evaluations_array.nbytes} bytes")
    
    # Feature analysis
    print(f"\n🔍 Feature Analysis:")
    print(f"   Non-zero features per position: {np.count_nonzero(features_array, axis=1).mean():.1f}")
    print(f"   Feature sparsity: {1.0 - np.count_nonzero(features_array)/features_array.size:.1%}")
    
    # Piece count analysis
    piece_names = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    max_pieces = [8, 2, 2, 2, 1, 1]
    
    for color_idx, color in enumerate(['White', 'Black']):
        print(f"\n♟️  {color} Piece Statistics:")
        base_idx = color_idx * 32
        
        for piece_idx, (piece_name, max_count) in enumerate(zip(piece_names, max_pieces)):
            start_idx = base_idx
            for prev_idx in range(piece_idx):
                start_idx += max_pieces[prev_idx] * 2
            
            # Count how many pieces of this type are present
            piece_counts = []
            for pos in range(len(features_array)):
                count = 0
                for slot in range(max_count):
                    coord_idx = start_idx + slot * 2
                    if features_array[pos, coord_idx] > 0:  # x coordinate > 0 means piece present
                        count += 1
                piece_counts.append(count)
            
            avg_count = np.mean(piece_counts)
            print(f"   {piece_name:6s}: {avg_count:.2f} average (max {max_count})")

def extract_dataset_features(fen_file: str, output_file: str, stockfish_path: str, 
                           time_limit: float = 0.1, analysis_depth: Optional[int] = None,
                           filter_extreme: bool = False, eval_threshold: float = 20.0) -> None:
    """
    Extract position features from FEN file and evaluate with Stockfish.
    
    Args:
        fen_file: Path to file containing FEN positions
        output_file: Output NPZ file path
        stockfish_path: Path to Stockfish executable
        time_limit: Time limit for Stockfish analysis in seconds
        analysis_depth: Optional depth limit for Stockfish analysis
        filter_extreme: Whether to filter out extreme evaluations
        eval_threshold: Threshold for extreme evaluation filtering (±pawns)
    """
    print(f"🔍 Extracting position features from {fen_file}")
    print(f"⚙️ Using Stockfish: {stockfish_path}")
    print(f"⏱️ Analysis time: {time_limit}s per position")
    if analysis_depth:
        print(f"🎯 Analysis depth: {analysis_depth}")
    if filter_extreme:
        print(f"🚫 Filtering evaluations > ±{eval_threshold}")
    
    # Initialize Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print(f"✅ Stockfish initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Stockfish: {e}")
        return
    
    # Read FEN positions
    with open(fen_file, 'r') as f:
        fens = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Loaded {len(fens)} FEN positions")
    
    features_list = []
    evaluations_list = []
    skipped_count = 0
    error_count = 0
    
    for i, fen in enumerate(fens):
        try:
            # Extract position features
            position_features = fen_to_position_features(fen)
            
            # Evaluate position with Stockfish
            board = chess.Board(fen)
            
            # Configure analysis parameters
            if analysis_depth:
                # Use depth limit
                result = engine.analyse(board, chess.engine.Limit(depth=analysis_depth))
            else:
                # Use time limit
                result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            
            score = result['score'].relative
            
            # Convert score to centipawns, handle mate scores
            if score.is_mate():
                # Convert mate in N to large evaluation
                mate_in = score.mate()
                if mate_in > 0:
                    eval_cp = 2000 - mate_in  # Mate for current side
                else:
                    eval_cp = -2000 - mate_in  # Mate against current side
            else:
                eval_cp = score.score()
            
            # Convert to pawn units
            evaluation = eval_cp / 100.0
            
            # Apply extreme evaluation filter if enabled
            if filter_extreme and abs(evaluation) > eval_threshold:
                skipped_count += 1
                if i % 100 == 0:
                    print(f"🚫 Skipped position {i+1}: evaluation {evaluation:+.2f} exceeds ±{eval_threshold}")
                continue
            
            features_list.append(position_features)
            evaluations_list.append(evaluation)
            
            # Progress update
            if (i + 1) % 100 == 0:
                processed = len(features_list)
                print(f"✅ Processed {i+1}/{len(fens)} positions, kept {processed} (skipped {skipped_count}, errors {error_count})")
                print(f"   Latest: eval={evaluation:+.2f}, features_shape={position_features.shape}")
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Show first few errors
                print(f"❌ Error processing position {i+1}: {e}")
            elif error_count == 6:
                print(f"❌ ... (suppressing further error messages)")
            continue
    
    # Close engine
    engine.quit()
    
    if not features_list:
        print(f"❌ No valid positions extracted!")
        return
    
    # Convert to numpy arrays
    features_array = np.array(features_list, dtype=np.float32)
    evaluations_array = np.array(evaluations_list, dtype=np.float32)
    
    print(f"\n📊 Feature Extraction Summary:")
    print(f"   Total positions processed: {len(fens)}")
    print(f"   Valid positions extracted: {len(features_list)}")
    print(f"   Positions skipped (extreme): {skipped_count}")
    print(f"   Positions with errors: {error_count}")
    print(f"   Success rate: {len(features_list)/len(fens)*100:.1f}%")
    print(f"   Feature vector size: {features_array.shape[1]}")
    print(f"   Evaluation range: {evaluations_array.min():+.2f} to {evaluations_array.max():+.2f}")
    
    # Save to NPZ file
    np.savez_compressed(output_file, 
                       features=features_array, 
                       evaluations=evaluations_array)
    
    print(f"💾 Saved {len(features_list)} positions to {output_file}")
    print(f"📏 Dataset size: {features_array.nbytes + evaluations_array.nbytes} bytes")
    
    # Feature analysis
    print(f"\n🔍 Feature Analysis:")
    print(f"   Non-zero features per position: {np.count_nonzero(features_array, axis=1).mean():.1f}")
    print(f"   Feature sparsity: {1.0 - np.count_nonzero(features_array)/features_array.size:.1%}")
    
    # Piece count analysis
    piece_names = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    max_pieces = [8, 2, 2, 2, 1, 1]
    
    for color_idx, color in enumerate(['White', 'Black']):
        print(f"\n♟️  {color} Piece Statistics:")
        base_idx = color_idx * 32
        
        for piece_idx, (piece_name, max_count) in enumerate(zip(piece_names, max_pieces)):
            start_idx = base_idx
            for prev_idx in range(piece_idx):
                start_idx += max_pieces[prev_idx] * 2
            
            # Count how many pieces of this type are present
            piece_counts = []
            for pos in range(len(features_array)):
                count = 0
                for slot in range(max_count):
                    coord_idx = start_idx + slot * 2
                    if features_array[pos, coord_idx] > 0:  # x coordinate > 0 means piece present
                        count += 1
                piece_counts.append(count)
            
            avg_count = np.mean(piece_counts)
            print(f"   {piece_name:6s}: {avg_count:.2f} average (max {max_count})")

def get_feature_names() -> List[str]:
    """Generate names for all 73 position features."""
    feature_names = []
    
    # Piece names and max counts
    piece_info = [
        ('Pawn', 8), ('Rook', 2), ('Knight', 2), 
        ('Bishop', 2), ('Queen', 1), ('King', 1)
    ]
    
    # Generate piece position feature names
    for color in ['White', 'Black']:
        for piece_name, max_count in piece_info:
            for slot in range(max_count):
                feature_names.append(f"{color}_{piece_name}_{slot+1}_x")
                feature_names.append(f"{color}_{piece_name}_{slot+1}_y")
    
    # Game state feature names
    game_state_names = [
        'white_to_move',
        'castle_K', 'castle_Q', 'castle_k', 'castle_q',
        'ep_a', 'ep_b', 'ep_c', 'ep_d'
    ]
    feature_names.extend(game_state_names)
    
    return feature_names

def main():
    """Main extraction pipeline."""
    parser = argparse.ArgumentParser(description='Extract position-based features from FEN or JSON')
    parser.add_argument('input_file', help='Input file (FEN text file or JSON evaluation file)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output NPZ file (default: auto-generated in dataset_pos_features/)')
    parser.add_argument('--stockfish', default='stockfish',
                       help='Path to Stockfish executable (only needed for FEN files)')
    parser.add_argument('--time', type=float, default=0.1,
                       help='Analysis time per position in seconds (FEN files only)')
    parser.add_argument('--depth', type=int, default=None,
                       help='Analysis depth limit (overrides time limit, FEN files only)')
    parser.add_argument('--filter-extreme', action='store_true',
                       help='Filter out extreme evaluations')
    parser.add_argument('--eval-threshold', type=float, default=20.0,
                       help='Threshold for extreme evaluation filtering in pawns (default: 20.0)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    # Determine file type and set default output
    is_json = input_path.suffix.lower() == '.json'
    
    if args.output is None:
        # Auto-generate output path in dataset_pos_features directory
        output_dir = Path("dataset_pos_features")
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename based on input
        base_name = input_path.stem
        if is_json:
            output_name = f"{base_name}_features.npz"
        else:
            output_name = f"{base_name}_fen_features.npz"
        
        args.output = str(output_dir / output_name)
    
    print("🚀 Position-based Feature Extractor")
    print("=" * 60)
    print(f"📂 Input: {args.input_file}")
    print(f"📊 Type: {'JSON evaluation file' if is_json else 'FEN text file'}")
    print(f"💾 Output: {args.output}")
    print(f"🎯 Feature format: 73 features (64 piece positions + 9 game state)")
    if args.filter_extreme:
        print(f"🚫 Filtering: evaluations > ±{args.eval_threshold}")
    print("=" * 60)
    
    # Extract features based on file type
    if is_json:
        # JSON evaluation file - no Stockfish needed
        extract_dataset_features_from_json(
            args.input_file, args.output, 
            args.filter_extreme, args.eval_threshold
        )
    else:
        # FEN text file - requires Stockfish evaluation
        extract_dataset_features(
            args.input_file, args.output, args.stockfish,
            args.time, args.depth, args.filter_extreme, args.eval_threshold
        )
    
    # Generate and save feature names
    feature_names = get_feature_names()
    feature_names_file = args.output.replace('.npz', '_feature_names.json')
    with open(feature_names_file, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"📝 Feature names saved to {feature_names_file}")
    print(f"✅ Position feature extraction complete!")
    print(f"🎯 Dataset ready for SR trainer!")
    
    return 0

if __name__ == "__main__":
    exit(main())