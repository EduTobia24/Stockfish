#!/usr/bin/env python3
"""
Chess Position Diagnostic Tool
=============================

Analyzes why some chess positions fail Stockfish evaluation.
Checks for illegal positions, parsing issues, and Stockfish problems.
"""

import subprocess
import time
from pathlib import Path

class ChessPositionDiagnostic:
    def __init__(self, stockfish_path: str = "src/stockfish"):
        self.stockfish_path = stockfish_path
    
    def validate_fen_syntax(self, fen: str) -> tuple[bool, str]:
        """Check if FEN string has valid syntax."""
        try:
            parts = fen.split(' ')
            if len(parts) != 6:
                return False, f"FEN must have 6 parts, found {len(parts)}"
            
            board_fen, turn, castling, en_passant, halfmove, fullmove = parts
            
            # Check board part
            ranks = board_fen.split('/')
            if len(ranks) != 8:
                return False, f"Board must have 8 ranks, found {len(ranks)}"
            
            for i, rank in enumerate(ranks):
                squares = 0
                for char in rank:
                    if char.isdigit():
                        squares += int(char)
                    elif char in 'prnbqkPRNBQK':
                        squares += 1
                    else:
                        return False, f"Invalid character '{char}' in rank {i+1}"
                
                if squares != 8:
                    return False, f"Rank {i+1} has {squares} squares, must be 8"
            
            # Check turn
            if turn not in ['w', 'b']:
                return False, f"Turn must be 'w' or 'b', found '{turn}'"
            
            # Check castling
            if not all(c in 'KQkq-' for c in castling):
                return False, f"Invalid castling rights: '{castling}'"
            
            # Check en passant
            if en_passant != '-':
                if len(en_passant) != 2 or en_passant[0] not in 'abcdefgh' or en_passant[1] not in '36':
                    return False, f"Invalid en passant square: '{en_passant}'"
            
            # Check move numbers
            try:
                half = int(halfmove)
                full = int(fullmove)
                if half < 0 or full < 1:
                    return False, f"Invalid move numbers: {halfmove}, {fullmove}"
            except ValueError:
                return False, f"Move numbers must be integers: '{halfmove}', '{fullmove}'"
            
            return True, "Valid FEN syntax"
            
        except Exception as e:
            return False, f"FEN parsing error: {str(e)}"
    
    def check_piece_count(self, fen: str) -> tuple[bool, str]:
        """Check if piece counts are reasonable."""
        board_fen = fen.split(' ')[0]
        
        # Count pieces
        piece_counts = {}
        for char in board_fen:
            if char in 'prnbqkPRNBQK':
                piece_counts[char] = piece_counts.get(char, 0) + 1
        
        # Check for exactly one king per side
        if piece_counts.get('K', 0) != 1:
            return False, f"Must have exactly 1 white king, found {piece_counts.get('K', 0)}"
        if piece_counts.get('k', 0) != 1:
            return False, f"Must have exactly 1 black king, found {piece_counts.get('k', 0)}"
        
        # Check for reasonable piece counts
        max_pieces = {'Q': 9, 'R': 10, 'B': 10, 'N': 10, 'P': 8}  # Including promoted pieces
        for piece in 'QRBNP':
            white_count = piece_counts.get(piece, 0)
            black_count = piece_counts.get(piece.lower(), 0)
            
            if white_count > max_pieces[piece]:
                return False, f"Too many white {piece}: {white_count} (max {max_pieces[piece]})"
            if black_count > max_pieces[piece]:
                return False, f"Too many black {piece.lower()}: {black_count} (max {max_pieces[piece]})"
        
        # Check total piece count
        total_pieces = sum(piece_counts.values())
        if total_pieces > 32:
            return False, f"Too many total pieces: {total_pieces} (max 32)"
        
        return True, "Piece counts valid"
    
    def test_stockfish_analysis(self, fen: str, timeout: int = 10) -> tuple[bool, str, dict]:
        """Test if Stockfish can analyze this position."""
        try:
            # Test with simple UCI commands
            commands = [
                "uci",
                "setoption name Hash value 32",
                "ucinewgame",
                f"position fen {fen}",
                "go depth 5",  # Quick shallow search
                "quit"
            ]
            
            input_text = '\n'.join(commands) + '\n'
            
            result = subprocess.run(
                [self.stockfish_path],
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return False, f"Stockfish returned error code {result.returncode}", {
                    'stdout': result.stdout[:500],
                    'stderr': result.stderr[:500]
                }
            
            # Check for error messages in output
            output = result.stdout.lower()
            if 'illegal' in output or 'invalid' in output or 'error' in output:
                return False, "Stockfish reported illegal/invalid position", {
                    'stdout': result.stdout[:1000],
                    'stderr': result.stderr[:500]
                }
            
            # Check if we got a bestmove
            if 'bestmove' not in result.stdout:
                return False, "No bestmove found in output", {
                    'stdout': result.stdout[:1000],
                    'stderr': result.stderr[:500]
                }
            
            # Parse analysis data
            analysis_data = self.parse_analysis_output(result.stdout)
            
            return True, "Stockfish analysis successful", analysis_data
            
        except subprocess.TimeoutExpired:
            return False, f"Stockfish analysis timed out after {timeout}s", {}
        except Exception as e:
            return False, f"Stockfish analysis error: {str(e)}", {}
    
    def parse_analysis_output(self, output: str) -> dict:
        """Parse Stockfish output for analysis data."""
        lines = output.strip().split('\n')
        
        analysis = {
            'best_move': None,
            'depth': 0,
            'evaluation': None,
            'nodes': 0,
            'info_lines': 0
        }
        
        for line in lines:
            if line.startswith('info') and 'depth' in line:
                analysis['info_lines'] += 1
                parts = line.split()
                
                if 'depth' in parts:
                    try:
                        depth_idx = parts.index('depth')
                        analysis['depth'] = max(analysis['depth'], int(parts[depth_idx + 1]))
                    except (ValueError, IndexError):
                        pass
                
                if 'score' in parts and 'cp' in parts:
                    try:
                        cp_idx = parts.index('cp')
                        analysis['evaluation'] = int(parts[cp_idx + 1]) / 100.0
                    except (ValueError, IndexError):
                        pass
                
                if 'nodes' in parts:
                    try:
                        nodes_idx = parts.index('nodes')
                        analysis['nodes'] = max(analysis['nodes'], int(parts[nodes_idx + 1]))
                    except (ValueError, IndexError):
                        pass
            
            elif line.startswith('bestmove'):
                parts = line.split()
                if len(parts) > 1:
                    analysis['best_move'] = parts[1]
        
        return analysis
    
    def diagnose_position(self, fen: str) -> dict:
        """Complete diagnostic of a chess position."""
        print(f"🔍 Diagnosing position: {fen}")
        
        result = {
            'fen': fen,
            'syntax_valid': False,
            'syntax_message': '',
            'pieces_valid': False,
            'pieces_message': '',
            'stockfish_valid': False,
            'stockfish_message': '',
            'stockfish_data': {},
            'overall_status': 'FAILED'
        }
        
        # Test 1: FEN syntax
        syntax_ok, syntax_msg = self.validate_fen_syntax(fen)
        result['syntax_valid'] = syntax_ok
        result['syntax_message'] = syntax_msg
        print(f"  📝 Syntax: {'✅' if syntax_ok else '❌'} {syntax_msg}")
        
        if not syntax_ok:
            return result
        
        # Test 2: Piece counts
        pieces_ok, pieces_msg = self.check_piece_count(fen)
        result['pieces_valid'] = pieces_ok
        result['pieces_message'] = pieces_msg
        print(f"  👑 Pieces: {'✅' if pieces_ok else '❌'} {pieces_msg}")
        
        if not pieces_ok:
            return result
        
        # Test 3: Stockfish analysis
        sf_ok, sf_msg, sf_data = self.test_stockfish_analysis(fen)
        result['stockfish_valid'] = sf_ok
        result['stockfish_message'] = sf_msg
        result['stockfish_data'] = sf_data
        print(f"  🔧 Stockfish: {'✅' if sf_ok else '❌'} {sf_msg}")
        
        if sf_ok:
            if sf_data.get('depth', 0) > 0:
                result['overall_status'] = 'SUCCESS'
                eval_str = f"{sf_data.get('evaluation', 0):+.2f}" if sf_data.get('evaluation') is not None else "N/A"
                print(f"       Analysis: Depth {sf_data.get('depth', 0)}, Eval {eval_str}, Move {sf_data.get('best_move', 'N/A')}")
            else:
                result['overall_status'] = 'PARTIAL'
                print(f"       Warning: Analysis succeeded but no depth achieved")
        
        return result

def test_problematic_positions():
    """Test some positions that might be problematic."""
    diagnostic = ChessPositionDiagnostic()
    
    print("🚀 Chess Position Diagnostic Tool")
    print("=" * 60)
    
    # Test cases
    test_positions = [
        # Valid starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        
        # Position with potential issues
        "r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 6",
        
        # Endgame position
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
        
        # Position with illegal king placement (both kings adjacent)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKKNR w KQkq - 0 1",
        
        # Position with too many queens
        "qqqqqqqq/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        
        # Malformed FEN (missing parts)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq",
        
        # Position where pawns are on wrong ranks
        "PPPPPPPP/rnbqkbnr/8/8/8/8/rnbqkbnr/PPPPPPPP w KQkq - 0 1"
    ]
    
    results = []
    for i, fen in enumerate(test_positions, 1):
        print(f"\n🧪 Test {i}/{len(test_positions)}:")
        result = diagnostic.diagnose_position(fen)
        results.append(result)
        print(f"   Status: {result['overall_status']}")
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r['overall_status'] == 'SUCCESS')
    partial_count = sum(1 for r in results if r['overall_status'] == 'PARTIAL')
    failed_count = sum(1 for r in results if r['overall_status'] == 'FAILED')
    
    print(f"✅ Successful: {success_count}/{len(results)}")
    print(f"⚠️  Partial: {partial_count}/{len(results)}")
    print(f"❌ Failed: {failed_count}/{len(results)}")
    
    # Show failed positions
    if failed_count > 0:
        print(f"\n❌ Failed Positions:")
        for r in results:
            if r['overall_status'] == 'FAILED':
                print(f"   FEN: {r['fen'][:50]}...")
                if not r['syntax_valid']:
                    print(f"      Syntax error: {r['syntax_message']}")
                elif not r['pieces_valid']:
                    print(f"      Piece error: {r['pieces_message']}")
                elif not r['stockfish_valid']:
                    print(f"      Stockfish error: {r['stockfish_message']}")

def diagnose_single_position(fen: str):
    """Diagnose a single position."""
    diagnostic = ChessPositionDiagnostic()
    result = diagnostic.diagnose_position(fen)
    
    print(f"\n📋 DETAILED DIAGNOSIS")
    print("=" * 60)
    print(f"FEN: {fen}")
    print(f"Status: {result['overall_status']}")
    
    if result['stockfish_data']:
        data = result['stockfish_data']
        print(f"\nStockfish Analysis:")
        print(f"  Best move: {data.get('best_move', 'N/A')}")
        print(f"  Depth: {data.get('depth', 0)}")
        print(f"  Evaluation: {data.get('evaluation', 'N/A')}")
        print(f"  Nodes: {data.get('nodes', 0):,}")
        print(f"  Info lines: {data.get('info_lines', 0)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose chess position evaluation issues')
    parser.add_argument('--fen', help='Test specific FEN position')
    parser.add_argument('--test-suite', action='store_true', help='Run test suite')
    
    args = parser.parse_args()
    
    if args.fen:
        diagnose_single_position(args.fen)
    elif args.test_suite:
        test_problematic_positions()
    else:
        print("Usage: python3 position_diagnostic.py --fen 'FEN_STRING' or --test-suite")

if __name__ == "__main__":
    main()