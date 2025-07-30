#!/usr/bin/env python3
"""
Script to extract and analyze criterion benchmark performance data
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_vector_size(dir_name: str) -> Tuple[str, int]:
    """Parse vector size directory name to get size name and bytes."""
    # Extract the numeric part and unit (KiB/MiB)
    match = re.match(r'Addition_(\d+\.?\d*)_([KM]iB)', dir_name)
    if not match:
        return dir_name, 0
    
    size_num = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    if unit == 'KiB':
        size_bytes = int(size_num * 1024)
    elif unit == 'MiB':
        size_bytes = int(size_num * 1024 * 1024)
    else:
        size_bytes = 0
    
    return dir_name, size_bytes


def calculate_throughput(time_ns: float, size_bytes: int) -> float:
    """Calculate throughput in GiB/s."""
    if time_ns == 0:
        return 0.0
    
    # Convert nanoseconds to seconds
    time_s = time_ns / 1_000_000_000
    
    # Calculate throughput: (size_in_bytes * 2) / (time_in_seconds * 1024^3)
    # Multiply by 2 because we're doing addition of two vectors
    throughput_gib_s = (size_bytes * 2) / (time_s * (1024 ** 3))
    
    return throughput_gib_s


def extract_benchmark_data(criterion_dir: str) -> List[Dict]:
    """Extract all benchmark data from criterion directory."""
    criterion_path = Path(criterion_dir)
    results = []
    
    # Find all Addition_* directories
    addition_dirs = sorted([d for d in criterion_path.iterdir() 
                           if d.is_dir() and d.name.startswith('Addition_')])
    
    for addition_dir in addition_dirs:
        vector_size_name, size_bytes = parse_vector_size(addition_dir.name)
        
        # Find algorithm directories (scalar, simd, ndarray, parallel_simd)
        algorithm_dirs = [d for d in addition_dir.iterdir() 
                         if d.is_dir() and d.name in ['scalar', 'simd', 'ndarray', 'parallel_simd']]
        
        for algo_dir in algorithm_dirs:
            algorithm = algo_dir.name
            
            # Find element count directories
            element_dirs = [d for d in algo_dir.iterdir() 
                           if d.is_dir() and d.name.isdigit()]
            
            for element_dir in element_dirs:
                element_count = int(element_dir.name)
                
                # Look for new/estimates.json
                estimates_file = element_dir / 'new' / 'estimates.json'
                
                if estimates_file.exists():
                    try:
                        with open(estimates_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract mean.point_estimate (time in nanoseconds)
                        time_ns = data['mean']['point_estimate']
                        
                        # Calculate throughput
                        throughput_gib_s = calculate_throughput(time_ns, size_bytes)
                        
                        result = {
                            'vector_size': vector_size_name,
                            'size_bytes': size_bytes,
                            'algorithm': algorithm,
                            'element_count': element_count,
                            'time_ns': time_ns,
                            'throughput_gib_s': throughput_gib_s
                        }
                        
                        results.append(result)
                        
                    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                        print(f"Error processing {estimates_file}: {e}")
    
    return results


def format_results(results: List[Dict]) -> str:
    """Format results in a structured way."""
    if not results:
        return "No benchmark data found."
    
    # Group by vector size
    by_vector_size = {}
    for result in results:
        vector_size = result['vector_size']
        if vector_size not in by_vector_size:
            by_vector_size[vector_size] = []
        by_vector_size[vector_size].append(result)
    
    output = []
    output.append("=== CRITERION BENCHMARK PERFORMANCE ANALYSIS ===\n")
    
    # Sort vector sizes by size_bytes
    sorted_vector_sizes = sorted(by_vector_size.keys(), 
                                key=lambda x: by_vector_size[x][0]['size_bytes'])
    
    for vector_size in sorted_vector_sizes:
        size_results = by_vector_size[vector_size]
        size_bytes = size_results[0]['size_bytes']
        
        output.append(f"Vector Size: {vector_size} ({size_bytes:,} bytes)")
        output.append("-" * 60)
        
        # Sort by algorithm and element count
        size_results.sort(key=lambda x: (x['algorithm'], x['element_count']))
        
        for result in size_results:
            output.append(f"  Algorithm: {result['algorithm']:<12} "
                         f"Elements: {result['element_count']:<8} "
                         f"Time: {result['time_ns']:<12,.2f} ns "
                         f"Throughput: {result['throughput_gib_s']:<8.2f} GiB/s")
        
        output.append("")
    
    # Summary statistics
    output.append("=== SUMMARY BY ALGORITHM ===\n")
    
    # Group by algorithm
    by_algorithm = {}
    for result in results:
        algo = result['algorithm']
        if algo not in by_algorithm:
            by_algorithm[algo] = []
        by_algorithm[algo].append(result)
    
    for algorithm in sorted(by_algorithm.keys()):
        algo_results = by_algorithm[algorithm]
        throughputs = [r['throughput_gib_s'] for r in algo_results]
        
        if throughputs:
            avg_throughput = sum(throughputs) / len(throughputs)
            max_throughput = max(throughputs)
            min_throughput = min(throughputs)
            
            output.append(f"{algorithm}:")
            output.append(f"  Measurements: {len(throughputs)}")
            output.append(f"  Avg Throughput: {avg_throughput:.2f} GiB/s")
            output.append(f"  Max Throughput: {max_throughput:.2f} GiB/s")
            output.append(f"  Min Throughput: {min_throughput:.2f} GiB/s")
            output.append("")
    
    return "\n".join(output)


def main():
    criterion_dir = "/home/mtantaoui/repos/mathlib/simdly/target/criterion"
    
    print("Extracting benchmark data...")
    results = extract_benchmark_data(criterion_dir)
    
    print(f"Found {len(results)} benchmark measurements")
    
    # Format and display results
    formatted_output = format_results(results)
    print("\n" + formatted_output)
    
    # Also save to a file
    output_file = "/home/mtantaoui/repos/mathlib/simdly/benchmark_analysis.txt"
    with open(output_file, 'w') as f:
        f.write(formatted_output)
    
    print(f"\nResults also saved to: {output_file}")


if __name__ == "__main__":
    main()