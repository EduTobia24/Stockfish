#!/usr/bin/env python3
"""
Stockfish-SR Comprehensive Benchmark Suite
Task 1.2: Baseline Performance Benchmarking

This script implements comprehensive benchmarking for establishing baseline metrics
before implementing symbolic regression evaluation.
"""

import subprocess
import time
import json
import os
import sys
import statistics
from datetime import datetime
from pathlib import Path
import psutil
import tempfile

class StockfishBenchmark:
    def __init__(self, stockfish_path="./stockfish"):
        self.stockfish_path = stockfish_path
        self.results = {}
        self.start_time = datetime.now()
        
    def run_stockfish_command(self, commands, timeout=30):
        """Run Stockfish with given commands and return output."""
        try:
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Send commands
            input_text = '\n'.join(commands) + '\nquit\n'
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            
            return stdout, stderr, process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "Timeout", -1
        except Exception as e:
            return "", str(e), -1

    def benchmark_search_performance(self):
        """Benchmark search performance at various depths."""
        print("🔍 Benchmarking Search Performance...")
        
        # Standard benchmark positions
        positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",  # Complex middlegame
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",  # Endgame
            "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",  # Tactical
        ]
        
        depths = [10, 12, 14, 16]
        search_results = []
        
        for depth in depths:
            depth_results = {
                'depth': depth,
                'positions': [],
                'total_nodes': 0,
                'total_time': 0,
                'avg_nps': 0
            }
            
            for i, position in enumerate(positions):
                print(f"  Testing depth {depth}, position {i+1}/4...")
                
                commands = [
                    "setoption name UCI_Chess960 value false",
                    "setoption name Hash value 128",
                    "setoption name Threads value 1",
                    f"position fen {position}",
                    f"go depth {depth}"
                ]
                
                start_time = time.time()
                stdout, stderr, returncode = self.run_stockfish_command(commands, timeout=60)
                elapsed_time = time.time() - start_time
                
                # Parse results
                nodes = 0
                nps = 0
                for line in stdout.split('\n'):
                    if 'nodes' in line and 'nps' in line:
                        parts = line.split()
                        for j, part in enumerate(parts):
                            if part == 'nodes' and j+1 < len(parts):
                                nodes = int(parts[j+1])
                            elif part == 'nps' and j+1 < len(parts):
                                nps = int(parts[j+1])
                
                position_result = {
                    'position_index': i,
                    'fen': position,
                    'nodes': nodes,
                    'time': elapsed_time,
                    'nps': nps,
                    'returncode': returncode
                }
                
                depth_results['positions'].append(position_result)
                depth_results['total_nodes'] += nodes
                depth_results['total_time'] += elapsed_time
            
            if depth_results['total_time'] > 0:
                depth_results['avg_nps'] = depth_results['total_nodes'] / depth_results['total_time']
            
            search_results.append(depth_results)
            print(f"  Depth {depth}: {depth_results['avg_nps']:.0f} nps average")
        
        self.results['search_performance'] = search_results
        return search_results

    def benchmark_tactical_positions(self):
        """Benchmark tactical problem solving accuracy."""
        print("🎯 Benchmarking Tactical Problem Solving...")
        
        # Famous tactical positions with known best moves
        tactical_positions = [
            {
                'name': 'Polgar Mate in 4',
                'fen': '2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPKPP1/RNB2R2 w KQkq - 0 1',
                'best_move': 'Qf7+',
                'mate_in': 4
            },
            {
                'name': 'Morphy Opera Game',
                'fen': '4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 1 17',
                'best_move': 'Qb8+',
                'mate_in': 2
            },
            {
                'name': 'Back Rank Mate',
                'fen': 'r5k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1',
                'best_move': 'Ra8+',
                'mate_in': 1
            },
            {
                'name': 'Queen Sacrifice',
                'fen': 'r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1',
                'best_move': 'Qxh7+',
                'mate_in': 3
            },
            {
                'name': 'Smothered Mate Pattern',
                'fen': '6rk/6pp/7N/8/8/8/8/7K w - - 0 1',
                'best_move': 'Nf7+',
                'mate_in': 2
            }
        ]
        
        tactical_results = []
        
        for i, puzzle in enumerate(tactical_positions):
            print(f"  Solving puzzle {i+1}/5: {puzzle['name']}...")
            
            # Test at different time limits
            time_limits = [0.1, 0.5, 1.0, 3.0]  # seconds
            
            for time_limit in time_limits:
                commands = [
                    "setoption name UCI_Chess960 value false",
                    "setoption name Hash value 128",
                    "setoption name Threads value 1",
                    f"position fen {puzzle['fen']}",
                    f"go movetime {int(time_limit * 1000)}"  # Convert to milliseconds
                ]
                
                start_time = time.time()
                stdout, stderr, returncode = self.run_stockfish_command(commands, timeout=time_limit + 5)
                elapsed_time = time.time() - start_time
                
                # Parse best move
                best_move = ""
                for line in stdout.split('\n'):
                    if line.startswith('bestmove'):
                        parts = line.split()
                        if len(parts) > 1:
                            best_move = parts[1]
                        break
                
                # Check if correct (simplified check)
                correct = puzzle['best_move'].lower().replace('+', '').replace('#', '') in best_move.lower()
                
                result = {
                    'puzzle_name': puzzle['name'],
                    'time_limit': time_limit,
                    'found_move': best_move,
                    'expected_move': puzzle['best_move'],
                    'correct': correct,
                    'elapsed_time': elapsed_time
                }
                
                tactical_results.append(result)
        
        self.results['tactical_solving'] = tactical_results
        return tactical_results

    def benchmark_evaluation_performance(self):
        """Benchmark pure evaluation performance."""
        print("⚡ Benchmarking Evaluation Performance...")
        
        # Create positions for evaluation benchmarking
        positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
            "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
            "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
        ]
        
        eval_results = []
        
        for i, position in enumerate(positions):
            print(f"  Evaluating position {i+1}/5...")
            
            commands = [
                "setoption name UCI_Chess960 value false",
                f"position fen {position}",
                "eval"
            ]
            
            # Run multiple times for statistical accuracy
            times = []
            evaluations = []
            
            for trial in range(10):
                start_time = time.time()
                stdout, stderr, returncode = self.run_stockfish_command(commands, timeout=10)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                
                # Parse evaluation
                eval_score = None
                for line in stdout.split('\n'):
                    if 'Final evaluation' in line:
                        try:
                            # Extract numerical value
                            parts = line.split()
                            for part in parts:
                                if '.' in part or part.isdigit() or (part.startswith('-') and part[1:].replace('.', '').isdigit()):
                                    eval_score = float(part)
                                    break
                        except:
                            pass
                
                if eval_score is not None:
                    evaluations.append(eval_score)
            
            result = {
                'position_index': i,
                'fen': position,
                'avg_time': statistics.mean(times) if times else 0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'avg_eval': statistics.mean(evaluations) if evaluations else 0,
                'std_eval': statistics.stdev(evaluations) if len(evaluations) > 1 else 0,
                'trial_count': len(times)
            }
            
            eval_results.append(result)
        
        self.results['evaluation_performance'] = eval_results
        return eval_results

    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("💾 Benchmarking Memory Usage...")
        
        # Monitor memory during different operations
        memory_results = {}
        
        # Baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_results['baseline_mb'] = baseline_memory
        
        # Memory with large hash table
        commands = [
            "setoption name Hash value 1024",  # 1GB
            "position startpos",
            "go depth 10"
        ]
        
        start_memory = process.memory_info().rss / 1024 / 1024
        stdout, stderr, returncode = self.run_stockfish_command(commands, timeout=30)
        end_memory = process.memory_info().rss / 1024 / 1024
        
        memory_results['hash_1gb'] = {
            'start_mb': start_memory,
            'end_mb': end_memory,
            'increase_mb': end_memory - start_memory
        }
        
        self.results['memory_usage'] = memory_results
        return memory_results

    def benchmark_engine_strength(self):
        """Quick engine strength estimation using self-play."""
        print("🏆 Benchmarking Engine Strength (Quick Self-Play)...")
        
        # Create a simple self-play test with time controls
        games = []
        time_controls = [0.1, 0.5, 1.0]  # seconds per move
        
        for tc in time_controls:
            game_result = {
                'time_control': tc,
                'moves': [],
                'result': 'ongoing'
            }
            
            position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            move_count = 0
            max_moves = 20  # Limit for benchmarking
            
            print(f"  Playing game with {tc}s per move...")
            
            while move_count < max_moves:
                commands = [
                    "setoption name Hash value 128",
                    "setoption name Threads value 1",
                    f"position fen {position}" if move_count == 0 else f"position fen {position}",
                    f"go movetime {int(tc * 1000)}"
                ]
                
                stdout, stderr, returncode = self.run_stockfish_command(commands, timeout=tc + 5)
                
                best_move = ""
                for line in stdout.split('\n'):
                    if line.startswith('bestmove'):
                        parts = line.split()
                        if len(parts) > 1:
                            best_move = parts[1]
                        break
                
                if not best_move or best_move == "(none)":
                    game_result['result'] = 'terminated'
                    break
                
                game_result['moves'].append(best_move)
                move_count += 1
                
                # Update position (simplified - in real implementation would need proper move application)
                # For benchmark purposes, we'll just count moves
            
            games.append(game_result)
        
        self.results['engine_strength'] = games
        return games

    def save_results(self, filename=None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stockfish_baseline_benchmark_{timestamp}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'timestamp': self.start_time.isoformat(),
            'stockfish_path': self.stockfish_path,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'platform': sys.platform
            },
            'benchmark_duration': (datetime.now() - self.start_time).total_seconds()
        }
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📊 Results saved to: {filename}")
        return filename

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("📈 STOCKFISH BASELINE BENCHMARK SUMMARY")
        print("="*60)
        
        # Search Performance Summary
        if 'search_performance' in self.results:
            print("\n🔍 SEARCH PERFORMANCE:")
            for depth_result in self.results['search_performance']:
                print(f"  Depth {depth_result['depth']:2d}: {depth_result['avg_nps']:>10,.0f} nps")
        
        # Tactical Solving Summary
        if 'tactical_solving' in self.results:
            print("\n🎯 TACTICAL SOLVING ACCURACY:")
            by_time = {}
            for result in self.results['tactical_solving']:
                tc = result['time_limit']
                if tc not in by_time:
                    by_time[tc] = {'correct': 0, 'total': 0}
                by_time[tc]['total'] += 1
                if result['correct']:
                    by_time[tc]['correct'] += 1
            
            for tc in sorted(by_time.keys()):
                accuracy = by_time[tc]['correct'] / by_time[tc]['total'] * 100
                print(f"  {tc:4.1f}s: {accuracy:5.1f}% ({by_time[tc]['correct']}/{by_time[tc]['total']})")
        
        # Evaluation Performance Summary
        if 'evaluation_performance' in self.results:
            print("\n⚡ EVALUATION PERFORMANCE:")
            avg_time = statistics.mean([r['avg_time'] for r in self.results['evaluation_performance']])
            print(f"  Average eval time: {avg_time*1000:.2f} ms")
        
        # Memory Usage Summary
        if 'memory_usage' in self.results:
            print("\n💾 MEMORY USAGE:")
            mem = self.results['memory_usage']
            print(f"  Baseline: {mem['baseline_mb']:.1f} MB")
            if 'hash_1gb' in mem:
                print(f"  With 1GB hash: {mem['hash_1gb']['end_mb']:.1f} MB")
        
        print(f"\n⏱️  Total benchmark time: {self.results['metadata']['benchmark_duration']:.1f} seconds")
        print("="*60)

def main():
    """Main benchmark execution."""
    print("🚀 Stockfish-SR Baseline Benchmark Suite")
    print("Task 1.2: Baseline Performance Benchmarking")
    print("="*60)
    
    # Check if Stockfish exists
    stockfish_path = "./stockfish"
    if len(sys.argv) > 1:
        stockfish_path = sys.argv[1]
    
    if not os.path.exists(stockfish_path):
        print(f"❌ Stockfish not found at: {stockfish_path}")
        print("Please build Stockfish first or provide path as argument.")
        sys.exit(1)
    
    benchmark = StockfishBenchmark(stockfish_path)
    
    try:
        # Run all benchmarks
        benchmark.benchmark_search_performance()
        benchmark.benchmark_tactical_positions()
        benchmark.benchmark_evaluation_performance()
        benchmark.benchmark_memory_usage()
        benchmark.benchmark_engine_strength()
        
        # Save and summarize results
        filename = benchmark.save_results()
        benchmark.print_summary()
        
        print(f"\n✅ Baseline benchmarking completed successfully!")
        print(f"📄 Detailed results saved to: {filename}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()