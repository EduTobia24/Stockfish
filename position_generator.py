#!/usr/bin/env python3
"""
Chess Position Generator
========================

Generates diverse chess positions using various methods without evaluation.
Fast position generation that can be evaluated later with different engines/settings.
"""

import chess
import chess.pgn
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Set
import io

class ChessPositionGenerator:
    """Generates diverse chess positions for training datasets."""
    
    def __init__(self):
        self.generated_positions = set()  # Track unique positions
        self.position_sources = {
            'random_moves': 0,
            'opening_book': 0,
            'endgame_patterns': 0,
            'tactical_positions': 0,
            'material_imbalances': 0
        }
    
    def generate_random_game_positions(self, num_positions: int = 100, 
                                     min_moves: int = 10, max_moves: int = 60) -> List[str]:
        """Generate positions from random games."""
        positions = []
        
        print(f"🎲 Generating {num_positions} positions from random games...")
        
        for i in range(num_positions * 3):  # Generate extra to account for duplicates
            if len(positions) >= num_positions:
                break
                
            board = chess.Board()
            num_moves = random.randint(min_moves, max_moves)
            
            try:
                # Play random moves
                for move_num in range(num_moves):
                    if board.is_game_over():
                        break
                    
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Bias towards more reasonable moves
                    if len(legal_moves) > 5:
                        # Prefer captures and checks
                        good_moves = [m for m in legal_moves 
                                    if board.is_capture(m) or board.gives_check(m)]
                        if good_moves and random.random() < 0.3:
                            move = random.choice(good_moves)
                        else:
                            move = random.choice(legal_moves)
                    else:
                        move = random.choice(legal_moves)
                    
                    board.push(move)
                
                # Only add if game not over and position is unique
                if not board.is_game_over():
                    fen = board.fen()
                    if fen not in self.generated_positions:
                        positions.append(fen)
                        self.generated_positions.add(fen)
                        self.position_sources['random_moves'] += 1
                        
                        if len(positions) % 50 == 0:
                            print(f"  Generated {len(positions)}/{num_positions} random positions...")
                            
            except Exception as e:
                continue  # Skip problematic positions
        
        return positions[:num_positions]
    
    def generate_opening_positions(self, num_positions: int = 100) -> List[str]:
        """Generate positions from common opening lines."""
        positions = []
        
        print(f"📚 Generating {num_positions} opening positions...")
        
        # Common opening move sequences
        opening_lines = [
            # Ruy Lopez
            ["e4", "e5", "Nf3", "Nc6", "Bb5"],
            ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"],
            ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"],
            
            # Sicilian
            ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"],
            ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4"],
            ["e4", "c5", "Nf3", "g6", "d4", "cxd4", "Nxd4"],
            
            # Queen's Gambit
            ["d4", "d5", "c4", "e6", "Nc3", "Nf6"],
            ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6"],
            ["d4", "d5", "c4", "c6", "Nf3", "Nf6"],
            
            # King's Indian
            ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"],
            ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"],
            
            # French Defense
            ["e4", "e6", "d4", "d5", "Nc3", "Bb4"],
            ["e4", "e6", "d4", "d5", "e5", "c5"],
            
            # English Opening
            ["c4", "e5", "Nc3", "Nf6", "g3", "d5"],
            ["c4", "c5", "Nc3", "Nc6", "g3", "g6"],
        ]
        
        for opening in opening_lines:
            for depth in range(len(opening)):
                try:
                    board = chess.Board()
                    
                    # Play the opening moves
                    for move_san in opening[:depth+1]:
                        move = board.parse_san(move_san)
                        board.push(move)
                    
                    # Generate variations by playing a few more moves
                    for _ in range(5):  # 5 variations per opening line
                        temp_board = board.copy()
                        
                        # Play 3-8 more moves randomly
                        for _ in range(random.randint(3, 8)):
                            if temp_board.is_game_over():
                                break
                            
                            legal_moves = list(temp_board.legal_moves)
                            if not legal_moves:
                                break
                            
                            move = random.choice(legal_moves)
                            temp_board.push(move)
                        
                        if not temp_board.is_game_over():
                            fen = temp_board.fen()
                            if fen not in self.generated_positions:
                                positions.append(fen)
                                self.generated_positions.add(fen)
                                self.position_sources['opening_book'] += 1
                                
                                if len(positions) >= num_positions:
                                    return positions[:num_positions]
                                    
                except Exception as e:
                    continue
        
        print(f"  Generated {len(positions)} opening positions")
        return positions[:num_positions]
    
    def generate_endgame_positions(self, num_positions: int = 50) -> List[str]:
        """Generate endgame positions with reduced material."""
        positions = []
        
        print(f"♔ Generating {num_positions} endgame positions...")
        
        for i in range(num_positions * 2):
            if len(positions) >= num_positions:
                break
            
            try:
                board = chess.Board()
                
                # Start with random game to reach endgame
                for _ in range(random.randint(40, 80)):
                    if board.is_game_over():
                        break
                    
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Bias towards exchanges in late game
                    if len(legal_moves) > 3:
                        capture_moves = [m for m in legal_moves if board.is_capture(m)]
                        if capture_moves and random.random() < 0.6:
                            move = random.choice(capture_moves)
                        else:
                            move = random.choice(legal_moves)
                    else:
                        move = random.choice(legal_moves)
                    
                    board.push(move)
                
                # Check if it's a proper endgame (few pieces)
                piece_count = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
                
                if 6 <= piece_count <= 12 and not board.is_game_over():
                    fen = board.fen()
                    if fen not in self.generated_positions:
                        positions.append(fen)
                        self.generated_positions.add(fen)
                        self.position_sources['endgame_patterns'] += 1
                        
            except Exception as e:
                continue
        
        print(f"  Generated {len(positions)} endgame positions")
        return positions[:num_positions]
    
    def generate_tactical_positions(self, num_positions: int = 50) -> List[str]:
        """Generate positions with tactical opportunities."""
        positions = []
        
        print(f"⚔️ Generating {num_positions} tactical positions...")
        
        for i in range(num_positions * 3):
            if len(positions) >= num_positions:
                break
            
            try:
                board = chess.Board()
                
                # Play to middlegame
                for _ in range(random.randint(15, 35)):
                    if board.is_game_over():
                        break
                    
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Prefer aggressive moves for tactics
                    aggressive_moves = []
                    for move in legal_moves:
                        if (board.is_capture(move) or 
                            board.gives_check(move) or
                            board.piece_at(move.from_square).piece_type in [chess.QUEEN, chess.ROOK]):
                            aggressive_moves.append(move)
                    
                    if aggressive_moves and random.random() < 0.4:
                        move = random.choice(aggressive_moves)
                    else:
                        move = random.choice(legal_moves)
                    
                    board.push(move)
                
                # Check for tactical elements
                if not board.is_game_over():
                    legal_moves = list(board.legal_moves)
                    has_tactics = any(board.is_capture(m) or board.gives_check(m) 
                                    for m in legal_moves)
                    
                    if has_tactics:
                        fen = board.fen()
                        if fen not in self.generated_positions:
                            positions.append(fen)
                            self.generated_positions.add(fen)
                            self.position_sources['tactical_positions'] += 1
                            
            except Exception as e:
                continue
        
        print(f"  Generated {len(positions)} tactical positions")
        return positions[:num_positions]
    
    def generate_material_imbalance_positions(self, num_positions: int = 50) -> List[str]:
        """Generate positions with material imbalances."""
        positions = []
        
        print(f"⚖️ Generating {num_positions} material imbalance positions...")
        
        for i in range(num_positions * 2):
            if len(positions) >= num_positions:
                break
            
            try:
                board = chess.Board()
                
                # Play game encouraging exchanges
                for _ in range(random.randint(20, 50)):
                    if board.is_game_over():
                        break
                    
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Strongly prefer captures to create imbalances
                    capture_moves = [m for m in legal_moves if board.is_capture(m)]
                    if capture_moves and random.random() < 0.8:
                        move = random.choice(capture_moves)
                    else:
                        move = random.choice(legal_moves)
                    
                    board.push(move)
                
                if not board.is_game_over():
                    # Calculate material balance
                    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                                  chess.ROOK: 5, chess.QUEEN: 9}
                    
                    white_material = sum(piece_values.get(board.piece_at(sq).piece_type, 0)
                                       for sq in chess.SQUARES 
                                       if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
                    
                    black_material = sum(piece_values.get(board.piece_at(sq).piece_type, 0)
                                       for sq in chess.SQUARES 
                                       if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
                    
                    # Keep positions with significant material imbalance
                    if abs(white_material - black_material) >= 2:
                        fen = board.fen()
                        if fen not in self.generated_positions:
                            positions.append(fen)
                            self.generated_positions.add(fen)
                            self.position_sources['material_imbalances'] += 1
                            
            except Exception as e:
                continue
        
        print(f"  Generated {len(positions)} material imbalance positions")
        return positions[:num_positions]
    
    def generate_diverse_dataset(self, total_positions: int = 1000) -> List[Dict]:
        """Generate a diverse dataset of chess positions."""
        
        print(f"🎯 Generating diverse dataset of {total_positions} positions...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Distribute positions across different types
        distribution = {
            'random_moves': int(total_positions * 0.4),      # 40%
            'opening_book': int(total_positions * 0.25),     # 25% 
            'tactical_positions': int(total_positions * 0.15), # 15%
            'endgame_patterns': int(total_positions * 0.1),   # 10%
            'material_imbalances': int(total_positions * 0.1)  # 10%
        }
        
        print(f"📊 Position distribution:")
        for pos_type, count in distribution.items():
            print(f"   {pos_type:20s}: {count:4d} positions")
        print()
        
        all_positions = []
        
        # Generate each type
        random_positions = self.generate_random_game_positions(distribution['random_moves'])
        opening_positions = self.generate_opening_positions(distribution['opening_book'])
        tactical_positions = self.generate_tactical_positions(distribution['tactical_positions'])
        endgame_positions = self.generate_endgame_positions(distribution['endgame_patterns'])
        imbalance_positions = self.generate_material_imbalance_positions(distribution['material_imbalances'])
        
        # Combine all positions
        position_sets = [
            (random_positions, 'random_moves'),
            (opening_positions, 'opening_book'),
            (tactical_positions, 'tactical_positions'),
            (endgame_positions, 'endgame_patterns'),
            (imbalance_positions, 'material_imbalances')
        ]
        
        for positions, pos_type in position_sets:
            for fen in positions:
                all_positions.append({
                    'fen': fen,
                    'position_type': pos_type,
                    'generated_at': time.time()
                })
        
        # Shuffle for variety
        random.shuffle(all_positions)
        
        # Limit to requested number
        all_positions = all_positions[:total_positions]
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Dataset generation complete!")
        print(f"📊 Generated {len(all_positions)} unique positions in {elapsed_time:.1f}s")
        print(f"📈 Generation rate: {len(all_positions)/elapsed_time:.1f} positions/second")
        print()
        print(f"🏷️ Final distribution:")
        for pos_type in self.position_sources:
            count = self.position_sources[pos_type]
            percentage = count / len(all_positions) * 100 if all_positions else 0
            print(f"   {pos_type:20s}: {count:4d} positions ({percentage:5.1f}%)")
        
        return all_positions

def main():
    """Main position generation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate diverse chess positions')
    parser.add_argument('--count', type=int, default=10000,
                       help='Number of positions to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='generated_positions',
                       help='Output filename prefix (default: generated_positions)')
    parser.add_argument('--format', choices=['json', 'txt'], default='json',
                       help='Output format (default: json)')
    
    args = parser.parse_args()
    
    print("🎲 Chess Position Generator")
    print("=" * 50)
    print(f"🎯 Target positions: {args.count}")
    print(f"📂 Output file: {args.output}.{args.format}")
    print("=" * 50)
    
    # Generate positions
    generator = ChessPositionGenerator()
    positions = generator.generate_diverse_dataset(args.count)
    
    # Save to file
    output_file = f"{args.output}.{args.format}"
    
    if args.format == 'json':
        with open(output_file, 'w') as f:
            json.dump(positions, f, indent=2)
    else:  # txt format
        with open(output_file, 'w') as f:
            for pos in positions:
                f.write(f"{pos['fen']}\\n")
    
    print(f"💾 Saved {len(positions)} positions to {output_file}")
    print(f"🎉 Position generation complete!")
    
    if args.format == 'json':
        print(f"\\n📋 Next step: Evaluate positions with:")
        print(f"   python3 stockfish_evaluator.py --input {output_file} --output {args.output}_evaluated")

if __name__ == "__main__":
    main()