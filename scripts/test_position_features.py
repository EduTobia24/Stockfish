#!/usr/bin/env python3
"""
Position Feature Test
====================

Test the position-based feature extractor with sample FEN positions.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from position_extractor import fen_to_position_features, get_feature_names
    import numpy as np
    import chess
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to install required packages:")
    print("   pip install python-chess numpy")
    exit(1)

def test_position_features():
    """Test position feature extraction with known positions."""
    
    print("🧪 Testing Position-based Feature Extractor")
    print("=" * 60)
    
    # Test positions
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Endgame position", "8/8/8/8/8/8/4K3/4k3 w - - 0 1"),
    ]
    
    feature_names = get_feature_names()
    print(f"📊 Feature vector size: {len(feature_names)}")
    print(f"   - Piece positions: 64 features (32 per color)")
    print(f"   - Game state: 9 features")
    print(f"   - Total: 73 features")
    print()
    
    for name, fen in test_positions:
        print(f"🔍 Testing: {name}")
        print(f"   FEN: {fen}")
        
        try:
            features = fen_to_position_features(fen)
            board = chess.Board(fen)
            
            print(f"   ✅ Features extracted: {features.shape}")
            print(f"   📈 Non-zero features: {np.count_nonzero(features)}")
            print(f"   🎯 Turn: {'White' if features[64] > 0.5 else 'Black'}")
            
            # Show piece positions
            print(f"   ♟️  Piece analysis:")
            
            # Analyze white pieces
            white_pieces = analyze_piece_positions(features, color='white')
            black_pieces = analyze_piece_positions(features, color='black')
            
            for piece, count in white_pieces.items():
                if count > 0:
                    print(f"      White {piece}: {count}")
            
            for piece, count in black_pieces.items():
                if count > 0:
                    print(f"      Black {piece}: {count}")
            
            # Show castling rights
            castling = []
            if features[65] > 0.5: castling.append('K')
            if features[66] > 0.5: castling.append('Q') 
            if features[67] > 0.5: castling.append('k')
            if features[68] > 0.5: castling.append('q')
            print(f"   🏰 Castling: {' '.join(castling) if castling else 'None'}")
            
            # Show en passant
            ep_files = ['a', 'b', 'c', 'd']
            ep_target = None
            for i, file_char in enumerate(ep_files):
                if features[69 + i] > 0.5:
                    ep_target = file_char
                    break
            print(f"   🎯 En passant: {ep_target if ep_target else 'None'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def analyze_piece_positions(features, color='white'):
    """Analyze piece positions from feature vector."""
    piece_names = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    max_pieces = [8, 2, 2, 2, 1, 1]
    
    base_idx = 0 if color == 'white' else 32
    piece_counts = {}
    
    for piece_idx, (piece_name, max_count) in enumerate(zip(piece_names, max_pieces)):
        start_idx = base_idx
        for prev_idx in range(piece_idx):
            start_idx += max_pieces[prev_idx] * 2
        
        count = 0
        for slot in range(max_count):
            coord_idx = start_idx + slot * 2
            if features[coord_idx] > 0:  # x coordinate > 0 means piece present
                count += 1
        
        piece_counts[piece_name] = count
    
    return piece_counts

def show_feature_mapping():
    """Show the complete feature mapping."""
    print("\n📋 Complete Feature Mapping (73 features):")
    print("=" * 60)
    
    feature_names = get_feature_names()
    
    for i, name in enumerate(feature_names):
        if i < 64:
            # Piece position features
            if i % 16 == 0:  # Start of new piece type
                color = "White" if i < 32 else "Black"
                piece_idx = (i % 32) // 16 if i < 32 else ((i - 32) % 32) // 16
                piece_names = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
                
                # Calculate correct piece index
                cumulative = 0
                for p_idx, max_count in enumerate([8, 2, 2, 2, 1, 1]):
                    if cumulative <= (i % 32) // 2 < cumulative + max_count:
                        piece_name = piece_names[p_idx]
                        break
                    cumulative += max_count
                
                print(f"\n   {color} {piece_name} positions:")
        
        print(f"     [{i:2d}] {name}")
    
    print(f"\n   Game state features:")
    for i in range(64, 73):
        print(f"     [{i:2d}] {feature_names[i]}")

if __name__ == "__main__":
    test_position_features()
    show_feature_mapping()