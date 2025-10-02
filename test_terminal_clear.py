#!/usr/bin/env python3
"""
Test Terminal Clearing
=====================

Quick test to verify terminal clearing works properly.
"""

import os
import time

def clear_terminal():
    """Clear the terminal screen for cleaner output."""
    try:
        # Try multiple methods for better compatibility
        if os.name == 'nt':  # Windows
            os.system('cls')
        else:  # Unix/Linux/Mac
            os.system('clear')
        
        # Also try ANSI escape sequences as backup
        print('\033[2J\033[H', end='', flush=True)
    except:
        # Fallback: print newlines
        print('\n' * 50)

def test_clearing():
    """Test the terminal clearing functionality."""
    
    print("🧪 Testing Terminal Clearing...")
    print("This is some initial text that should disappear.")
    print("More text here...")
    print("And even more text...")
    
    print("\n⏳ Clearing in 3 seconds...")
    time.sleep(3)
    
    clear_terminal()
    
    print("✅ Terminal cleared!")
    print("If you can only see this message and the one above,")
    print("then terminal clearing is working correctly!")
    
    print("\n🔄 Testing multiple clears...")
    time.sleep(2)
    
    for i in range(3):
        clear_terminal()
        print(f"🎯 Clear test #{i+1}")
        print("This should replace the previous message")
        time.sleep(1.5)
    
    clear_terminal()
    print("🎉 Terminal clearing test complete!")
    print("✅ All clears worked if you only see this final message")

if __name__ == "__main__":
    test_clearing()