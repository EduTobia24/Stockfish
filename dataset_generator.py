#!/usr/bin/env python3
"""
Stockfish Dataset Generator
===========================

Generates a dataset of chess positions with their Stockfish evaluations.
Each position gets 5 seconds of analysis to produce high-quality evaluations.
"""

import subprocess
import time
import json
import random
import csv
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chess
import chess.pgn
import io

class StockfishDatasetGenerator:
    def __init__(self, stockfish_path: str = "src/stockfish"):
        self.stockfish_path = stockfish_path
        self.dataset = []
        
        # Position sources for diversity
        self.starting_positions = [
            # Standard opening positions
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            
            # Common opening setups
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
            
            # Sicilian Defense variations
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            
            # French Defense
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            
            # Caro-Kann Defense  
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ]

    def verify_stockfish(self) -> bool:
        """Verify Stockfish is working"""
        try:
            result = subprocess.run(
                [self.stockfish_path], 
                input='uci\nquit\n',
                text=True, 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0 and "uciok" in result.stdout
        except:
            return False

    def analyze_position(self, fen: str, move_time_seconds: int = 5) -> Optional[Dict]:
        """Analyze a position with Stockfish for specified time"""
        try:
            # UCI commands for analysis
            commands = [
                "uci",
                f"position fen {fen}",
                f"go movetime {move_time_seconds * 1000}",  # Convert to milliseconds
                "quit"
            ]
            
            result = subprocess.run(
                [self.stockfish_path],
                input='\n'.join(commands) + '\n',
                text=True,
                capture_output=True,
                timeout=move_time_seconds + 10  # Extra buffer
            )
            
            if result.returncode != 0:
                return None
            
            # Parse the analysis output
            analysis = self.parse_stockfish_output(result.stdout, fen)
            return analysis
            
        except Exception as e:
            print(f"    ❌ Analysis failed: {e}")
            return None

    def parse_stockfish_output(self, output: str, fen: str) -> Dict:
        """Parse Stockfish output to extract evaluation data"""
        lines = output.strip().split('\n')
        
        # Initialize analysis data
        analysis = {
            'fen': fen,
            'evaluation': None,
            'best_move': None,
            'depth': 0,
            'nodes': 0,
            'nps': 0,
            'time_ms': 0,
            'pv': [],
            'mate_in': None
        }
        
        best_depth = 0
        
        for line in lines:
            # Parse info lines with search data
            if line.startswith('info') and 'depth' in line:
                parts = line.split()
                try:
                    # Extract depth
                    if 'depth' in parts:
                        depth = int(parts[parts.index('depth') + 1])
                        if depth > best_depth:
                            best_depth = depth
                            analysis['depth'] = depth
                    
                    # Extract evaluation
                    if 'score' in parts:
                        score_idx = parts.index('score')
                        if score_idx + 2 < len(parts):
                            score_type = parts[score_idx + 1]
                            score_value = int(parts[score_idx + 2])
                            
                            if score_type == 'cp':
                                # Centipawn evaluation
                                analysis['evaluation'] = score_value / 100.0
                            elif score_type == 'mate':
                                # Mate in N moves
                                analysis['mate_in'] = score_value
                                analysis['evaluation'] = 999.0 if score_value > 0 else -999.0
                    
                    # Extract nodes
                    if 'nodes' in parts:
                        analysis['nodes'] = int(parts[parts.index('nodes') + 1])
                    
                    # Extract NPS
                    if 'nps' in parts:
                        analysis['nps'] = int(parts[parts.index('nps') + 1])
                    
                    # Extract time
                    if 'time' in parts:
                        analysis['time_ms'] = int(parts[parts.index('time') + 1])
                    
                    # Extract principal variation
                    if 'pv' in parts:
                        pv_idx = parts.index('pv')
                        analysis['pv'] = parts[pv_idx + 1:]
                        
                except (ValueError, IndexError):
                    continue
            
            # Parse best move
            elif line.startswith('bestmove'):
                parts = line.split()
                if len(parts) > 1:
                    analysis['best_move'] = parts[1]
        
        return analysis

    def generate_random_position(self, base_fen: str, max_moves: int = 15) -> str:
        """Generate a random position by playing random moves from base position"""
        try:
            board = chess.Board(base_fen)
            
            # Play random moves
            moves_played = 0
            while moves_played < max_moves and not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                    
                # Prefer captures and checks occasionally
                if random.random() < 0.3:
                    priority_moves = [move for move in legal_moves 
                                    if board.is_capture(move) or board.gives_check(move)]
                    if priority_moves:
                        legal_moves = priority_moves
                
                move = random.choice(legal_moves)
                board.push(move)
                moves_played += 1
                
                # Stop if position becomes too one-sided
                if moves_played > 5:
                    # Quick evaluation to avoid completely lost positions
                    try:
                        quick_analysis = self.analyze_position(board.fen(), 1)
                        if quick_analysis and quick_analysis.get('evaluation'):
                            if abs(quick_analysis['evaluation']) > 8.0:  # Too one-sided
                                break
                    except:
                        pass
            
            return board.fen()
        except:
            return base_fen

    def generate_dataset(self, target_positions: int = 1000, analysis_time: int = 5) -> List[Dict]:
        """Generate dataset of positions with evaluations"""
        print(f"🎯 Generating dataset of {target_positions} positions")
        print(f"⏱️  Analysis time: {analysis_time} seconds per position")
        print("=" * 60)
        
        if not self.verify_stockfish():
            print("❌ Stockfish verification failed!")
            return []
        
        positions_generated = 0
        failed_analyses = 0
        
        print(f"📍 Starting position generation...")
        
        while positions_generated < target_positions:
            # Select random starting position
            base_fen = random.choice(self.starting_positions)
            
            # Generate random variation
            random_moves = random.randint(0, 20)
            position_fen = self.generate_random_position(base_fen, random_moves)
            
            print(f"  Position {positions_generated + 1}/{target_positions}: ", end="")
            
            # Analyze with Stockfish
            analysis = self.analyze_position(position_fen, analysis_time)
            
            if analysis and analysis.get('evaluation') is not None:
                # Add metadata
                analysis['position_id'] = positions_generated + 1
                analysis['generation_method'] = 'random_play'
                analysis['base_position'] = base_fen
                analysis['analysis_time_seconds'] = analysis_time
                
                self.dataset.append(analysis)
                positions_generated += 1
                
                eval_str = f"{analysis['evaluation']:+.2f}"
                if analysis.get('mate_in'):
                    eval_str = f"M{analysis['mate_in']}"
                
                print(f"✅ Eval: {eval_str}, Depth: {analysis['depth']}, "
                      f"Best: {analysis.get('best_move', 'N/A')}")
                
                # Save progress every 50 positions
                if positions_generated % 50 == 0:
                    self.save_dataset(f"dataset_progress_{positions_generated}.json")
                    print(f"    💾 Progress saved at {positions_generated} positions")
                    
            else:
                failed_analyses += 1
                print(f"❌ Analysis failed")
                
                # If too many failures, something is wrong
                if failed_analyses > 20:
                    print(f"\n⚠️  Too many analysis failures ({failed_analyses}). Stopping.")
                    break
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📊 Generated {len(self.dataset)} positions")
        print(f"❌ Failed analyses: {failed_analyses}")
        
        return self.dataset

    def save_dataset(self, filename: str = "stockfish_dataset.json") -> str:
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        return filename

    def save_dataset_csv(self, filename: str = "stockfish_dataset.csv") -> str:
        """Save dataset to CSV file for easy analysis"""
        if not self.dataset:
            return filename
            
        fieldnames = [
            'position_id', 'fen', 'evaluation', 'mate_in', 'best_move', 
            'depth', 'nodes', 'nps', 'time_ms', 'pv_moves'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for analysis in self.dataset:
                row = {
                    'position_id': analysis.get('position_id', ''),
                    'fen': analysis.get('fen', ''),
                    'evaluation': analysis.get('evaluation', ''),
                    'mate_in': analysis.get('mate_in', ''),
                    'best_move': analysis.get('best_move', ''),
                    'depth': analysis.get('depth', 0),
                    'nodes': analysis.get('nodes', 0),
                    'nps': analysis.get('nps', 0),
                    'time_ms': analysis.get('time_ms', 0),
                    'pv_moves': ' '.join(analysis.get('pv', [])[:5])  # First 5 moves of PV
                }
                writer.writerow(row)
        
        return filename

    def print_dataset_summary(self):
        """Print summary statistics of the generated dataset"""
        if not self.dataset:
            print("📊 No dataset to summarize")
            return
        
        print("\n" + "=" * 60)
        print("📊 DATASET SUMMARY")
        print("=" * 60)
        
        # Basic stats
        total_positions = len(self.dataset)
        evaluations = [pos['evaluation'] for pos in self.dataset if pos.get('evaluation') is not None]
        mate_positions = [pos for pos in self.dataset if pos.get('mate_in') is not None]
        
        print(f"📍 Total positions: {total_positions}")
        print(f"⚖️  Positions with evaluations: {len(evaluations)}")
        print(f"👑 Mate positions: {len(mate_positions)}")
        
        if evaluations:
            print(f"\n📈 Evaluation Statistics:")
            print(f"   Average: {sum(evaluations) / len(evaluations):+.2f}")
            print(f"   Range: {min(evaluations):+.2f} to {max(evaluations):+.2f}")
            
            # Distribution
            white_advantage = sum(1 for e in evaluations if e > 0.5)
            black_advantage = sum(1 for e in evaluations if e < -0.5)
            balanced = len(evaluations) - white_advantage - black_advantage
            
            print(f"   White advantage (>+0.5): {white_advantage} ({white_advantage/len(evaluations)*100:.1f}%)")
            print(f"   Balanced (±0.5): {balanced} ({balanced/len(evaluations)*100:.1f}%)")
            print(f"   Black advantage (<-0.5): {black_advantage} ({black_advantage/len(evaluations)*100:.1f}%)")
        
        # Depth statistics
        depths = [pos.get('depth', 0) for pos in self.dataset]
        if depths:
            print(f"\n🎯 Search Depth Statistics:")
            print(f"   Average depth: {sum(depths) / len(depths):.1f}")
            print(f"   Depth range: {min(depths)} to {max(depths)}")
        
        # Performance statistics
        nodes_list = [pos.get('nodes', 0) for pos in self.dataset if pos.get('nodes', 0) > 0]
        if nodes_list:
            avg_nodes = sum(nodes_list) / len(nodes_list)
            print(f"\n⚡ Performance Statistics:")
            print(f"   Average nodes analyzed: {avg_nodes:,.0f}")
            
        nps_list = [pos.get('nps', 0) for pos in self.dataset if pos.get('nps', 0) > 0]
        if nps_list:
            avg_nps = sum(nps_list) / len(nps_list)
            print(f"   Average nodes per second: {avg_nps:,.0f}")
        
        print("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Stockfish evaluation dataset')
    parser.add_argument('--stockfish', default='src/stockfish', 
                       help='Path to Stockfish executable')
    parser.add_argument('--positions', type=int, default=1000,
                       help='Number of positions to generate')
    parser.add_argument('--time', type=int, default=5,
                       help='Analysis time per position in seconds')
    parser.add_argument('--output', default='stockfish_dataset',
                       help='Output filename prefix (without extension)')
    
    args = parser.parse_args()
    
    # Check if Stockfish exists
    if not Path(args.stockfish).exists():
        print(f"❌ Stockfish not found at {args.stockfish}")
        return 1
    
    print("🚀 Stockfish Dataset Generator")
    print("=" * 60)
    print(f"🔧 Stockfish: {args.stockfish}")
    print(f"🎯 Target positions: {args.positions}")
    print(f"⏱️  Analysis time: {args.time}s per position")
    print(f"💾 Output prefix: {args.output}")
    print("=" * 60)
    
    # Generate dataset
    generator = StockfishDatasetGenerator(args.stockfish)
    dataset = generator.generate_dataset(args.positions, args.time)
    
    if dataset:
        # Save in multiple formats
        json_file = generator.save_dataset(f"{args.output}.json")
        csv_file = generator.save_dataset_csv(f"{args.output}.csv")
        
        print(f"\n💾 Dataset saved:")
        print(f"   JSON: {json_file}")
        print(f"   CSV:  {csv_file}")
        
        # Print summary
        generator.print_dataset_summary()
        
        print(f"\n✅ Dataset generation completed successfully!")
        return 0
    else:
        print(f"\n❌ Dataset generation failed!")
        return 1

if __name__ == "__main__":
    exit(main())