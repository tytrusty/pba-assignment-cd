#!/usr/bin/env python3
"""
Script to run all test scenes and generate USD output files.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run all test scenes and generate USD outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for simulation (default: cpu)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for USD files (default: ./output)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of simulation steps (default: 100)"
    )
    args = parser.parse_args()
    
    # Get the tests directory
    script_dir = Path(__file__).parent
    tests_dir = script_dir / "tests"
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all test files (Python files excluding __init__.py)
    test_files = sorted(tests_dir.glob("*.py"))
    test_files = [f for f in test_files if f.name != "__init__.py"]
    
    if not test_files:
        print(f"No test files found in {tests_dir}")
        return 1
    
    print(f"Found {len(test_files)} test files")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Device: {args.device}")
    print(f"Number of steps: {args.num_steps}")
    print("-" * 60)
    
    # Run each test
    for i, test_file in enumerate(test_files, 1):
        test_name = test_file.stem  # e.g., "bunny_fall" from "bunny_fall.py"
        usd_output = output_dir / f"{test_name}.usd"
        
        print(f"\n[{i}/{len(test_files)}] Running {test_file.name}...")
        print(f"  Output: {usd_output}")
        
        # Build command
        cmd = [
            sys.executable,
            "main.py",
            "--scene", str(test_file),
            "--usd_output", str(usd_output),
            "--num_steps", str(args.num_steps),
            "--device", args.device
        ]
        
        # Run the command
        subprocess.run(
            cmd,
            cwd=script_dir,
            check=True
        )
    print("\n" + "=" * 60)
    print(f"All {len(test_files)} tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

