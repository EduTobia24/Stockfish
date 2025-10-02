#!/usr/bin/env python3
"""
Stockfish Position Evaluator
============================

Evaluates chess positions using Stockfish engine.
Takes position files and outputs evaluations with detailed analysis.
Includes incremental saving to prevent data loss.
"""

import chess
import chess.engine
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class StockfishEvaluator:
    """Evaluates chess positions using Stockfish engine."""
    
    def __init__(self, stockfish_path: str = None, threads: int = 4, 
                 hash_size: int = 128, analysis_time: float = 5.0):
        """Initialize Stockfish evaluator."""
        
        self.stockfish_path = stockfish_path or self.find_stockfish()
        self.threads = threads
        self.hash_size = hash_size
        self.analysis_time = analysis_time
        self.engine = None
        
        self.stats = {
            'total_positions': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_time_per_position': 0.0,
            'failure_reasons': {}
        }
    
    def find_stockfish(self) -> str:
        """Find Stockfish executable in common locations."""
        
        possible_paths = [
            './stockfish',
            './src/stockfish', 
            'stockfish',
            '/usr/bin/stockfish',
            '/usr/local/bin/stockfish',
            'C:\\stockfish\\stockfish.exe',
            'stockfish.exe'
        ]
        
        for path in possible_paths:
            try:
                # Test if stockfish works
                result = subprocess.run([path, '--help'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    print(f"🔍 Found Stockfish at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        raise FileNotFoundError("❌ Stockfish not found. Please provide path with --stockfish-path")
    
    def start_engine(self):
        """Start the Stockfish engine."""
        
        if self.engine is not None:
            return  # Already started
        
        try:
            print(f"🚀 Starting Stockfish engine...")
            print(f"   Path: {self.stockfish_path}")
            print(f"   Threads: {self.threads}")
            print(f"   Hash: {self.hash_size}MB")
            print(f"   Analysis time: {self.analysis_time}s per position")
            
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            
            # Check available options
            available_options = list(self.engine.options.keys())
            print(f"🔍 Stockfish supports {len(available_options)} options")
            
            # Configure engine with only supported options
            config = {}
            
            # Add options only if they're supported
            if "Threads" in available_options:
                config["Threads"] = self.threads
            if "Hash" in available_options:
                config["Hash"] = self.hash_size
            if "Move Overhead" in available_options:
                config["Move Overhead"] = 100
            
            if config:
                self.engine.configure(config)
                print(f"✅ Configured options: {list(config.keys())}")
            else:
                print(f"⚠️ No configurable options found, using defaults")
            
            print(f"✅ Stockfish engine started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start Stockfish: {e}")
            raise
    
    def stop_engine(self):
        """Stop the Stockfish engine."""
        
        if self.engine:
            try:
                self.engine.quit()
                self.engine = None
                print(f"🛑 Stockfish engine stopped")
            except Exception as e:
                print(f"⚠️ Error stopping engine: {e}")
    
    def evaluate_position(self, fen: str) -> Optional[Dict]:
        """Evaluate a single chess position."""
        
        try:
            # Parse FEN
            board = chess.Board(fen)
            
            if board.is_game_over():
                return {
                    'fen': fen,
                    'evaluation': None,
                    'depth': 0,
                    'nodes': 0,
                    'time': 0.0,
                    'best_move': None,
                    'pv': [],
                    'status': 'game_over',
                    'error': 'Position is game over'
                }
            
            start_time = time.time()
            
            # Analyze position
            analysis = self.engine.analyse(
                board, 
                chess.engine.Limit(time=self.analysis_time),
                multipv=1
            )
            
            evaluation_time = time.time() - start_time
            
            # Extract evaluation
            score = analysis.get('score')
            if score is None:
                return {
                    'fen': fen,
                    'evaluation': None,
                    'depth': analysis.get('depth', 0),
                    'nodes': analysis.get('nodes', 0),
                    'time': evaluation_time,
                    'best_move': str(analysis.get('pv', [None])[0]) if analysis.get('pv') else None,
                    'pv': [str(move) for move in analysis.get('pv', [])],
                    'status': 'no_score',
                    'error': 'No score in analysis'
                }
            
            # Convert score to centipawns from white's perspective
            if score.is_mate():
                # Mate scores: positive for white advantage, negative for black
                mate_in = score.white().mate()
                if mate_in > 0:
                    evaluation = 10000 - mate_in  # White mates
                else:
                    evaluation = -10000 - mate_in  # Black mates
            else:
                # Regular centipawn scores
                evaluation = score.white().score() / 100.0  # Convert to pawns
            
            return {
                'fen': fen,
                'evaluation': evaluation,
                'depth': analysis.get('depth', 0),
                'nodes': analysis.get('nodes', 0),
                'time': evaluation_time,
                'best_move': str(analysis.get('pv', [None])[0]) if analysis.get('pv') else None,
                'pv': [str(move) for move in analysis.get('pv', [])[:5]],  # First 5 moves
                'status': 'success',
                'error': None
            }
            
        except ValueError:  # chess library raises ValueError for invalid FEN
            return {
                'fen': fen,
                'evaluation': None,
                'depth': 0,
                'nodes': 0,
                'time': 0.0,
                'best_move': None,
                'pv': [],
                'status': 'invalid_fen',
                'error': 'Invalid FEN string'
            }
            
        except Exception as e:
            return {
                'fen': fen,
                'evaluation': None,
                'depth': 0,
                'nodes': 0,
                'time': 0.0,
                'best_move': None,
                'pv': [],
                'status': 'engine_error',
                'error': str(e)
            }
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to file (JSON or CSV based on extension)."""
        
        try:
            if output_file.endswith('.json'):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif output_file.endswith('.csv'):
                import csv
                if results:
                    with open(output_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
            return True
        except Exception as e:
            print(f"   ⚠️ Save failed: {e}")
            return False
    
    def evaluate_positions_from_file(self, input_file: str, output_file: str = None, 
                                   save_interval: int = 50) -> List[Dict]:
        """Evaluate positions from input file with incremental saving."""
        
        print(f"📂 Loading positions from {input_file}")
        
        # Load positions
        if input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # List of position objects or FEN strings
                if isinstance(data[0], dict):
                    positions = [(pos.get('fen', pos), pos) for pos in data]
                else:
                    positions = [(fen, {'fen': fen}) for fen in data]
            else:
                raise ValueError("JSON file should contain a list of positions")
                
        else:  # Assume text file with one FEN per line
            with open(input_file, 'r') as f:
                fens = [line.strip() for line in f if line.strip()]
            positions = [(fen, {'fen': fen}) for fen in fens]
        
        print(f"✅ Loaded {len(positions)} positions")
        if output_file:
            print(f"💾 Will save progress every {save_interval} positions to {output_file}")
        
        # Start engine
        self.start_engine()
        
        # Evaluate positions
        results = []
        start_time = time.time()
        
        try:
            for i, (fen, original_data) in enumerate(positions):
                self.stats['total_positions'] += 1
                
                # Show current position being evaluated
                position_type = original_data.get('position_type', 'unknown')
                print(f"\\r🔍 Evaluating position {i+1}/{len(positions)}: {position_type} - {fen[:50]}{'...' if len(fen) > 50 else ''}", end='', flush=True)
                
                # Evaluate position
                evaluation_result = self.evaluate_position(fen)
                
                # Combine with original data
                result = {**original_data, **evaluation_result}
                results.append(result)
                
                # Update stats
                if evaluation_result['status'] == 'success':
                    self.stats['successful_evaluations'] += 1
                else:
                    self.stats['failed_evaluations'] += 1
                    failure_reason = evaluation_result['status']
                    self.stats['failure_reasons'][failure_reason] = \\
                        self.stats['failure_reasons'].get(failure_reason, 0) + 1
                
                # Progress update every 10 positions
                if (i + 1) % 10 == 0 or (i + 1) == len(positions):
                    print("\\r" + " " * 120 + "\\r", end='')  # Clear the line
                    
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(positions) - i - 1) / rate if rate > 0 else 0
                    
                    success_rate = self.stats['successful_evaluations'] / (i + 1) * 100
                    
                    # Show detailed progress
                    print(f"📊 Progress: {i+1:4d}/{len(positions)} "
                          f"({success_rate:5.1f}% success, "
                          f"{rate:4.1f} pos/s, "
                          f"ETA: {remaining:3.0f}s)")
                    
                    # Show last evaluation result
                    if evaluation_result['status'] == 'success':
                        eval_val = evaluation_result['evaluation']
                        depth = evaluation_result['depth']
                        best_move = evaluation_result['best_move']
                        print(f"   ✅ Last eval: {eval_val:+.2f} (depth {depth}, best: {best_move})")
                    else:
                        error = evaluation_result['error'][:50] if evaluation_result['error'] else 'Unknown'
                        print(f"   ❌ Last eval failed: {error}")
                
                # Incremental save every save_interval positions
                if output_file and (i + 1) % save_interval == 0:
                    if self.save_results(results, output_file):
                        print(f"   💾 Saved {len(results)} results to {output_file}")
        
        finally:
            # Clear any remaining position display
            print("\\r" + " " * 120 + "\\r", end='', flush=True)
            
            # Stop engine
            self.stop_engine()
        
        # Calculate final stats
        total_time = time.time() - start_time
        self.stats['average_time_per_position'] = total_time / len(positions)
        
        print(f"\\n✅ Evaluation complete!")
        self.print_stats()
        
        return results
    
    def print_stats(self):
        """Print evaluation statistics."""
        
        print(f"\\n📊 Evaluation Statistics:")
        print(f"   Total positions: {self.stats['total_positions']}")
        print(f"   Successful: {self.stats['successful_evaluations']} "
              f"({self.stats['successful_evaluations']/self.stats['total_positions']*100:.1f}%)")
        print(f"   Failed: {self.stats['failed_evaluations']} "
              f"({self.stats['failed_evaluations']/self.stats['total_positions']*100:.1f}%)")
        print(f"   Average time per position: {self.stats['average_time_per_position']:.3f}s")
        
        if self.stats['failure_reasons']:
            print(f"\\n❌ Failure breakdown:")
            for reason, count in self.stats['failure_reasons'].items():
                print(f"   {reason}: {count}")

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate chess positions with Stockfish')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file with positions (JSON or text)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for evaluations')
    parser.add_argument('--stockfish-path', type=str, default=None,
                       help='Path to Stockfish executable')
    parser.add_argument('--time', type=float, default=5.0,
                       help='Analysis time per position in seconds (default: 5.0)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads for Stockfish (default: 4)')
    parser.add_argument('--hash', type=int, default=128,
                       help='Hash table size in MB (default: 128)')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='Save progress every N positions (default: 50)')
    
    args = parser.parse_args()
    
    print("🐟 Stockfish Position Evaluator")
    print("=" * 50)
    print(f"📂 Input: {args.input}")
    print(f"💾 Output: {args.output}.{args.format}")
    print(f"⏱️  Analysis time: {args.time}s per position")
    print(f"🧵 Threads: {args.threads}")
    print(f"💾 Hash: {args.hash}MB")
    print(f"💾 Save every: {args.save_interval} positions")
    print("=" * 50)
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create evaluator
    try:
        evaluator = StockfishEvaluator(
            stockfish_path=args.stockfish_path,
            threads=args.threads,
            hash_size=args.hash,
            analysis_time=args.time
        )
    except Exception as e:
        print(f"❌ Failed to create evaluator: {e}")
        return 1
    
    # Evaluate positions
    try:
        output_file_with_ext = f"{args.output}.{args.format}"
        results = evaluator.evaluate_positions_from_file(
            args.input, 
            output_file_with_ext,
            args.save_interval
        )
        
        # Final save (ensures all results are saved)
        evaluator.save_results(results, output_file_with_ext)
        
        print(f"\\n💾 Final save: {len(results)} evaluations to {output_file_with_ext}")
        
        # Show sample results
        successful_results = [r for r in results if r['status'] == 'success']
        if successful_results:
            print(f"\\n🔍 Sample evaluations:")
            for i, result in enumerate(successful_results[:5]):
                print(f"   {i+1}. Eval: {result['evaluation']:+.2f}, "
                      f"Depth: {result['depth']}, "
                      f"Best: {result['best_move']}")
        
        print(f"\\n🎉 Evaluation complete!")
        print(f"📋 Next step: Extract bitboard features with:")
        print(f"   python3 bitboard_extractor.py --dataset {output_file_with_ext} --output extracted_features")
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())