#!/usr/bin/env python3
"""
Final Stockfish Dataset Generator  
=================================

Generates a dataset with proper real-time output capture from Stockfish.
Uses time-based search for consistent evaluation quality.
"""

import subprocess
import time
import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Optional
import threading
import queue

class FinalStockfishDatasetGenerator:
    def __init__(self, stockfish_path: str = "src/stockfish"):
        self.stockfish_path = stockfish_path
        self.dataset = []
        
        # Rich set of chess positions
        self.positions = [
            # Starting positions and early game
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            
            # Popular openings
            "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
            "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 4",
            
            # Middlegame positions
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
            "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
            "rq3rk1/ppp2ppp/1bnpN3/3N2B1/4P3/7P/PPPQ1PP1/2KR3R b - - 0 14",
            "r1bq1r1k/1pp1npp1/p2p3p/1b6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
            "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/3PP3/2PBPN2/PP3PPP/RNBQ1RK1 w - - 4 8",
            "r1bqr1k1/pp1n1ppp/2p1pn2/3p4/2PP4/1PB1PN2/P3BPPP/R2Q1RK1 b - - 0 10",
            
            # Endgame positions
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
            "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
            "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
            "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 4 3",
            "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
            "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
            
            # Tactical positions
            "r1bbk1nr/pp3p1p/2nppqp1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 8",
            "r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 6",
            "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            
            # Complex strategic positions
            "rnbq1rk1/ppp1ppbp/3p1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQK2R w KQ - 0 6",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 2 4",
            "2rq1rk1/1p1bppbp/p2p1np1/8/3PP3/1BP2N1P/PP2BPP1/R2Q1RK1 w - - 0 12",
            "r1bq1rk1/pp2ppbp/2np1np1/8/3PP3/2N1BN2/PPP1BPPP/R2Q1RK1 w - - 0 8",
            "r1bqr1k1/pp1n1ppp/2pb1n2/3p4/3P4/1BP1PN2/PP2BPPP/R2Q1RK1 w - - 0 10",
            
            # Imbalanced material positions 
            "r1b1k2r/ppqn1ppp/2pbpn2/8/3P4/2NBPN2/PPQ2PPP/R1B1K2R w KQkq - 0 8",
            "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 10",
            "rnb1k1nr/pp1p1ppp/4p3/q7/1bPP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5"
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

    def analyze_position_interactive(self, fen: str, time_limit: int = 5) -> Optional[Dict]:
        """Analyze position with interactive communication to capture all output"""
        try:
            # Start Stockfish process
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Send UCI initialization
            process.stdin.write("uci\n")
            process.stdin.write("setoption name Hash value 128\n")
            process.stdin.write("setoption name Threads value 1\n")
            process.stdin.write("ucinewgame\n")
            process.stdin.write(f"position fen {fen}\n")
            process.stdin.write(f"go movetime {time_limit * 1000}\n")
            process.stdin.flush()
            
            # Collect output
            output_lines = []
            start_time = time.time()
            
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                    
                output_lines.append(line.strip())
                
                # Stop when we get bestmove or timeout
                if line.startswith('bestmove') or (time.time() - start_time) > time_limit + 5:
                    break
            
            # Cleanup
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.terminate()
            process.wait(timeout=2)
            
            # Parse collected output
            output_text = '\n'.join(output_lines)
            return self.parse_stockfish_output(output_text, fen)
            
        except Exception as e:
            return None

    def parse_stockfish_output(self, output: str, fen: str) -> Dict:
        """Parse Stockfish output to extract evaluation data"""
        lines = output.strip().split('\n')
        
        analysis = {
            'fen': fen,
            'evaluation': 0.0,
            'best_move': None,
            'depth': 0,
            'nodes': 0,
            'nps': 0,
            'time_ms': 0,
            'pv': [],
            'mate_in': None,
            'raw_output': output[:500] + "..." if len(output) > 500 else output  # Debug info
        }
        
        # Find the best (final) analysis
        best_depth = 0
        final_eval = None
        
        for line in lines:
            if line.startswith('info') and ('depth' in line or 'score' in line):
                parts = line.split()
                try:
                    # Extract depth if present
                    current_depth = 0
                    if 'depth' in parts:
                        depth_idx = parts.index('depth')
                        if depth_idx + 1 < len(parts):
                            current_depth = int(parts[depth_idx + 1])
                    
                    # Process if this is our best depth so far
                    if current_depth >= best_depth and 'score' in parts:
                        best_depth = current_depth
                        analysis['depth'] = current_depth
                        
                        # Extract score
                        score_idx = parts.index('score')
                        if score_idx + 2 < len(parts):
                            score_type = parts[score_idx + 1]
                            score_value = int(parts[score_idx + 2])
                            
                            if score_type == 'cp':
                                final_eval = score_value / 100.0
                                analysis['mate_in'] = None
                            elif score_type == 'mate':
                                analysis['mate_in'] = score_value
                                final_eval = 999.0 if score_value > 0 else -999.0
                        
                        # Extract other information
                        if 'nodes' in parts:
                            nodes_idx = parts.index('nodes')
                            if nodes_idx + 1 < len(parts):
                                analysis['nodes'] = int(parts[nodes_idx + 1])
                        
                        if 'nps' in parts:
                            nps_idx = parts.index('nps')
                            if nps_idx + 1 < len(parts):
                                analysis['nps'] = int(parts[nps_idx + 1])
                        
                        if 'time' in parts:
                            time_idx = parts.index('time')
                            if time_idx + 1 < len(parts):
                                analysis['time_ms'] = int(parts[time_idx + 1])
                        
                        if 'pv' in parts:
                            pv_idx = parts.index('pv')
                            analysis['pv'] = parts[pv_idx + 1:]
                        
                except (ValueError, IndexError):
                    continue
            
            elif line.startswith('bestmove'):
                parts = line.split()
                if len(parts) > 1:
                    analysis['best_move'] = parts[1]
        
        if final_eval is not None:
            analysis['evaluation'] = final_eval
        
        return analysis

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
        
        # Randomize order for variety
        available_positions = self.positions.copy()
        random.shuffle(available_positions)
        
        position_index = 0
        
        while positions_generated < target_positions:
            # Get next position (cycle through available ones)
            base_fen = available_positions[position_index % len(available_positions)]
            
            # Add small variations 20% of the time
            if random.random() < 0.2:
                position_fen = self.create_variation(base_fen)
            else:
                position_fen = base_fen
                
            position_index += 1
            
            print(f"  Position {positions_generated + 1}/{target_positions}: ", end="", flush=True)
            
            # Analyze the position
            analysis = self.analyze_position_interactive(position_fen, analysis_time)
            
            if (analysis and 
                analysis.get('best_move') and 
                analysis.get('best_move') != 'none' and
                analysis.get('evaluation') is not None):
                
                # Add metadata
                analysis['position_id'] = positions_generated + 1
                analysis['analysis_time_seconds'] = analysis_time
                analysis['generation_method'] = 'interactive_analysis'
                
                # Remove raw output from final dataset (was just for debugging)
                if 'raw_output' in analysis:
                    del analysis['raw_output']
                
                self.dataset.append(analysis)
                positions_generated += 1
                
                eval_str = f"{analysis['evaluation']:+.2f}"
                if analysis.get('mate_in'):
                    eval_str = f"M{analysis['mate_in']}"
                
                depth_info = f"D{analysis.get('depth', 0)}"
                nodes_info = f"{analysis.get('nodes', 0):,}n" if analysis.get('nodes', 0) > 0 else "0n"
                
                print(f"✅ {eval_str} | {depth_info} | {nodes_info} | {analysis.get('best_move', 'N/A')}")
                
                # Save progress every 50 positions
                if positions_generated % 50 == 0:
                    self.save_dataset(f"dataset_progress_{positions_generated}.json")
                    print(f"    💾 Progress saved")
                    
            else:
                failed_analyses += 1
                depth = analysis.get('depth', 0) if analysis else 0
                print(f"❌ Failed (d{depth})")
                
                if failed_analyses > 100:
                    print(f"\n⚠️  Too many failures ({failed_analyses}). Stopping.")
                    break
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📊 Generated {len(self.dataset)} positions")
        print(f"❌ Failed analyses: {failed_analyses}")
        
        return self.dataset

    def create_variation(self, base_fen: str) -> str:
        """Create minor variations"""
        parts = base_fen.split(' ')
        if len(parts) >= 3:
            # Modify castling rights
            castling = parts[2]
            if castling != '-' and len(castling) > 1:
                chars = list(castling)
                if len(chars) > 1:
                    chars.pop(random.randint(0, len(chars) - 1))
                    parts[2] = ''.join(chars) if chars else '-'
                    return ' '.join(parts)
        return base_fen

    def save_dataset(self, filename: str) -> str:
        """Save dataset to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        return filename

    def save_dataset_csv(self, filename: str) -> str:
        """Save dataset to CSV"""
        if not self.dataset:
            return filename
            
        fieldnames = [
            'position_id', 'fen', 'evaluation', 'mate_in', 'best_move', 
            'depth', 'nodes', 'nps', 'time_ms', 'pv_line'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pos in self.dataset:
                row = {
                    'position_id': pos.get('position_id', ''),
                    'fen': pos.get('fen', ''),
                    'evaluation': pos.get('evaluation', ''),
                    'mate_in': pos.get('mate_in', ''),
                    'best_move': pos.get('best_move', ''),
                    'depth': pos.get('depth', 0),
                    'nodes': pos.get('nodes', 0),
                    'nps': pos.get('nps', 0),
                    'time_ms': pos.get('time_ms', 0),
                    'pv_line': ' '.join(pos.get('pv', [])[:6])
                }
                writer.writerow(row)
        
        return filename

    def print_summary(self):
        """Print dataset summary"""
        if not self.dataset:
            print("📊 No dataset generated")
            return
        
        print("\n" + "=" * 60)
        print("📊 DATASET SUMMARY")
        print("=" * 60)
        
        total = len(self.dataset)
        evals = [p['evaluation'] for p in self.dataset if p.get('evaluation') is not None]
        depths = [p.get('depth', 0) for p in self.dataset]
        nodes = [p.get('nodes', 0) for p in self.dataset if p.get('nodes', 0) > 0]
        mate_pos = [p for p in self.dataset if p.get('mate_in') is not None]
        
        print(f"📍 Total positions: {total}")
        print(f"👑 Mate positions: {len(mate_pos)}")
        
        if evals:
            avg_eval = sum(evals) / len(evals)
            print(f"\n📈 Evaluations:")
            print(f"   Average: {avg_eval:+.2f}")
            print(f"   Range: {min(evals):+.2f} to {max(evals):+.2f}")
            
            white_wins = sum(1 for e in evals if e > 1.0)
            balanced = sum(1 for e in evals if -1.0 <= e <= 1.0)
            black_wins = sum(1 for e in evals if e < -1.0)
            
            print(f"   White advantage (>+1.0): {white_wins}")
            print(f"   Balanced (±1.0): {balanced}")
            print(f"   Black advantage (<-1.0): {black_wins}")
        
        if depths:
            print(f"\n🎯 Search depths: {min(depths)}-{max(depths)} (avg: {sum(depths)/len(depths):.1f})")
        
        if nodes:
            print(f"⚡ Avg nodes: {sum(nodes)/len(nodes):,.0f}")
        
        print("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Stockfish evaluation dataset')
    parser.add_argument('--stockfish', default='src/stockfish', help='Stockfish path')
    parser.add_argument('--positions', type=int, default=1000, help='Number of positions')
    parser.add_argument('--time', type=int, default=5, help='Analysis time per position (seconds)')
    parser.add_argument('--output', default='stockfish_dataset', help='Output filename prefix')
    
    args = parser.parse_args()
    
    if not Path(args.stockfish).exists():
        print(f"❌ Stockfish not found at {args.stockfish}")
        return 1
    
    print("🚀 Final Stockfish Dataset Generator")
    print("=" * 60)
    print(f"🔧 Stockfish: {args.stockfish}")
    print(f"🎯 Target positions: {args.positions}")
    print(f"⏱️  Analysis time: {args.time}s per position")
    print(f"💾 Output prefix: {args.output}")
    print("=" * 60)
    
    # Generate dataset
    generator = FinalStockfishDatasetGenerator(args.stockfish)
    dataset = generator.generate_dataset(args.positions, args.time)
    
    if dataset:
        # Save files
        json_file = generator.save_dataset(f"{args.output}.json")
        csv_file = generator.save_dataset_csv(f"{args.output}.csv")
        
        print(f"\n💾 Files saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📊 CSV:  {csv_file}")
        
        generator.print_summary()
        print(f"\n✅ SUCCESS! Dataset with {len(dataset)} positions ready for training.")
        return 0
    else:
        print(f"\n❌ Dataset generation failed!")
        return 1

if __name__ == "__main__":
    exit(main())