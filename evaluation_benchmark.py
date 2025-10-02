#!/usr/bin/env python3
"""
Stockfish Evaluation Benchmark Tool
Focused on evaluation function performance and accuracy
"""

import subprocess
import time
import json
import statistics
import re
from pathlib import Path
from tactical_test_suite import *

class EvaluationBenchmark:
    def __init__(self, stockfish_path="./stockfish"):
        self.stockfish_path = stockfish_path
        self.results = {}
    
    def run_stockfish_eval(self, fen, timeout=10):
        """Get evaluation for a specific position."""
        try:
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            commands = [
                f"position fen {fen}",
                "eval",
                "quit"
            ]
            
            input_text = '\n'.join(commands) + '\n'
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            
            return stdout, stderr
        except Exception as e:
            return "", str(e)
    
    def parse_evaluation(self, stdout):
        """Parse evaluation output to extract numerical score."""
        eval_score = None
        eval_details = {}
        
        lines = stdout.split('\n')
        for line in lines:
            if 'Final evaluation' in line:
                # Extract numerical evaluation
                match = re.search(r'Final evaluation\s+([+-]?\d*\.?\d+)', line)
                if match:
                    eval_score = float(match.group(1))
            
            # Extract NNUE components if available
            if 'NNUE evaluation' in line:
                match = re.search(r'NNUE evaluation\s+([+-]?\d*\.?\d+)', line)
                if match:
                    eval_details['nnue_raw'] = float(match.group(1))
        
        return eval_score, eval_details
    
    def benchmark_evaluation_speed(self):
        """Benchmark evaluation speed across different position types."""
        print("⚡ Benchmarking Evaluation Speed...")
        
        speed_results = {}
        
        # Test different position complexities
        test_sets = {
            'simple': [pos['fen'] for pos in PERFORMANCE_POSITIONS if pos['complexity'] == 'low'],
            'medium': [pos['fen'] for pos in PERFORMANCE_POSITIONS if pos['complexity'] == 'medium'],
            'complex': [pos['fen'] for pos in PERFORMANCE_POSITIONS if pos['complexity'] == 'high']
        }
        
        for complexity, positions in test_sets.items():
            print(f"  Testing {complexity} positions...")
            
            times = []
            for fen in positions:
                # Run multiple trials
                for trial in range(20):
                    start_time = time.perf_counter()
                    stdout, stderr = self.run_stockfish_eval(fen)
                    end_time = time.perf_counter()
                    
                    elapsed = end_time - start_time
                    times.append(elapsed)
            
            if times:
                speed_results[complexity] = {
                    'avg_time_ms': statistics.mean(times) * 1000,
                    'std_time_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'evaluations_per_second': 1.0 / statistics.mean(times) if statistics.mean(times) > 0 else 0,
                    'trial_count': len(times)
                }
        
        self.results['evaluation_speed'] = speed_results
        return speed_results
    
    def benchmark_evaluation_consistency(self):
        """Test evaluation consistency across multiple runs."""
        print("🔄 Benchmarking Evaluation Consistency...")
        
        consistency_results = []
        
        # Select a few key positions
        test_positions = [
            ('Starting Position', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
            ('Complex Middlegame', 'r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9'),
            ('Endgame', '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11')
        ]
        
        for name, fen in test_positions:
            print(f"  Testing consistency for: {name}")
            
            evaluations = []
            for trial in range(50):  # Many trials for statistical significance
                stdout, stderr = self.run_stockfish_eval(fen)
                eval_score, details = self.parse_evaluation(stdout)
                
                if eval_score is not None:
                    evaluations.append(eval_score)
            
            if evaluations:
                consistency_result = {
                    'position_name': name,
                    'fen': fen,
                    'mean_eval': statistics.mean(evaluations),
                    'std_eval': statistics.stdev(evaluations) if len(evaluations) > 1 else 0,
                    'min_eval': min(evaluations),
                    'max_eval': max(evaluations),
                    'range_eval': max(evaluations) - min(evaluations),
                    'trial_count': len(evaluations),
                    'is_consistent': statistics.stdev(evaluations) < 0.01 if len(evaluations) > 1 else True
                }
                consistency_results.append(consistency_result)
        
        self.results['evaluation_consistency'] = consistency_results
        return consistency_results
    
    def benchmark_evaluation_accuracy(self):
        """Test evaluation accuracy on known positions."""
        print("🎯 Benchmarking Evaluation Accuracy...")
        
        accuracy_results = []
        
        # Test on tactical positions with known outcomes
        test_categories = [
            ('Mate in 1', MATE_IN_1, 'mate'),
            ('Mate in 2', MATE_IN_2, 'mate'),
            ('Winning Tactics', TACTICAL_MOTIFS, 'advantage'),
            ('Balanced Middlegames', MIDDLEGAME_POSITIONS, 'balanced')
        ]
        
        for category_name, positions, expected_type in test_categories:
            print(f"  Testing {category_name}...")
            
            category_results = []
            
            for pos in positions:
                stdout, stderr = self.run_stockfish_eval(pos['fen'])
                eval_score, details = self.parse_evaluation(stdout)
                
                if eval_score is not None:
                    # Determine if evaluation matches expectation
                    correct_eval = False
                    
                    if expected_type == 'mate':
                        # Should show large advantage or mate score
                        correct_eval = abs(eval_score) > 5.0
                    elif expected_type == 'advantage':
                        # Should show some advantage
                        correct_eval = abs(eval_score) > 0.5
                    elif expected_type == 'balanced':
                        # Should be roughly balanced
                        correct_eval = abs(eval_score) < 1.0
                    
                    result = {
                        'position_name': pos['name'],
                        'fen': pos['fen'],
                        'evaluation': eval_score,
                        'expected_type': expected_type,
                        'correct_evaluation': correct_eval,
                        'details': details
                    }
                    category_results.append(result)
            
            accuracy_results.append({
                'category': category_name,
                'results': category_results,
                'accuracy_rate': sum(1 for r in category_results if r['correct_evaluation']) / len(category_results) if category_results else 0
            })
        
        self.results['evaluation_accuracy'] = accuracy_results
        return accuracy_results
    
    def benchmark_evaluation_scaling(self):
        """Test evaluation behavior across different material balances."""
        print("⚖️ Benchmarking Evaluation Scaling...")
        
        # Create positions with known material differences
        scaling_positions = [
            {
                'name': 'Material Equality',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'material_diff': 0
            },
            {
                'name': 'Pawn Up',
                'fen': 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2',
                'material_diff': 0  # Traded pawns
            },
            {
                'name': 'Minor Piece Up',
                'fen': 'rnbqk2r/ppppppbp/5np1/8/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 4 4',
                'material_diff': 0  # Equal minor pieces
            }
        ]
        
        scaling_results = []
        
        for pos in scaling_positions:
            stdout, stderr = self.run_stockfish_eval(pos['fen'])
            eval_score, details = self.parse_evaluation(stdout)
            
            if eval_score is not None:
                result = {
                    'position_name': pos['name'],
                    'fen': pos['fen'],
                    'material_difference': pos['material_diff'],
                    'evaluation': eval_score,
                    'eval_per_material': eval_score / max(1, abs(pos['material_diff'])) if pos['material_diff'] != 0 else eval_score
                }
                scaling_results.append(result)
        
        self.results['evaluation_scaling'] = scaling_results
        return scaling_results
    
    def generate_report(self):
        """Generate comprehensive evaluation benchmark report."""
        print("\n" + "="*70)
        print("📊 STOCKFISH EVALUATION BENCHMARK REPORT")
        print("="*70)
        
        # Speed Report
        if 'evaluation_speed' in self.results:
            print("\n⚡ EVALUATION SPEED:")
            speed = self.results['evaluation_speed']
            for complexity, metrics in speed.items():
                print(f"  {complexity.capitalize():8s}: {metrics['avg_time_ms']:6.2f} ms avg, {metrics['evaluations_per_second']:8.0f} eval/sec")
        
        # Consistency Report
        if 'evaluation_consistency' in self.results:
            print("\n🔄 EVALUATION CONSISTENCY:")
            for result in self.results['evaluation_consistency']:
                status = "✅ Consistent" if result['is_consistent'] else "⚠️  Variable"
                print(f"  {result['position_name']:20s}: {status} (std: {result['std_eval']:.4f})")
        
        # Accuracy Report
        if 'evaluation_accuracy' in self.results:
            print("\n🎯 EVALUATION ACCURACY:")
            for category in self.results['evaluation_accuracy']:
                accuracy = category['accuracy_rate'] * 100
                print(f"  {category['category']:20s}: {accuracy:5.1f}% accurate")
        
        # Scaling Report
        if 'evaluation_scaling' in self.results:
            print("\n⚖️ EVALUATION SCALING:")
            for result in self.results['evaluation_scaling']:
                print(f"  {result['position_name']:20s}: {result['evaluation']:+6.2f}")
        
        print("\n" + "="*70)
    
    def save_detailed_results(self, filename=None):
        """Save detailed results to JSON."""
        if filename is None:
            filename = "evaluation_benchmark_detailed.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        return filename

def main():
    """Run evaluation benchmark."""
    print("🔍 Stockfish Evaluation Function Benchmark")
    print("="*50)
    
    benchmark = EvaluationBenchmark()
    
    try:
        # Run all evaluation benchmarks
        benchmark.benchmark_evaluation_speed()
        benchmark.benchmark_evaluation_consistency()
        benchmark.benchmark_evaluation_accuracy()
        benchmark.benchmark_evaluation_scaling()
        
        # Generate reports
        benchmark.generate_report()
        benchmark.save_detailed_results()
        
        print("\n✅ Evaluation benchmarking completed!")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())