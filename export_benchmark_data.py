#!/usr/bin/env python3
"""
Export benchmark data in CSV and JSON formats for further analysis
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List
import re


def extract_benchmark_data(criterion_dir: str) -> List[Dict]:
    """Extract all benchmark data from criterion directory."""
    criterion_path = Path(criterion_dir)
    results = []
    
    # Find all Addition_* directories
    addition_dirs = sorted([d for d in criterion_path.iterdir() 
                           if d.is_dir() and d.name.startswith('Addition_')])
    
    for addition_dir in addition_dirs:
        # Parse vector size
        match = re.match(r'Addition_(\d+\.?\d*)_([KM]iB)', addition_dir.name)
        if not match:
            continue
            
        size_num = float(match.group(1))
        unit = match.group(2)
        
        # Convert to bytes
        if unit == 'KiB':
            size_bytes = int(size_num * 1024)
        elif unit == 'MiB':
            size_bytes = int(size_num * 1024 * 1024)
        else:
            continue
        
        # Find algorithm directories
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
                        
                        time_ns = data['mean']['point_estimate']
                        time_s = time_ns / 1_000_000_000
                        throughput_gib_s = (size_bytes * 2) / (time_s * (1024 ** 3))
                        
                        # Calculate additional metrics
                        throughput_gb_s = (size_bytes * 2) / (time_s * (1000 ** 3))  # GB/s (decimal)
                        elements_per_second = element_count / time_s
                        bytes_per_element = size_bytes / element_count if element_count > 0 else 0
                        
                        result = {
                            'vector_size_name': addition_dir.name,
                            'size_display': f"{size_num} {unit}",
                            'size_bytes': size_bytes,
                            'size_kb': size_bytes / 1024,
                            'size_mb': size_bytes / (1024 * 1024),
                            'algorithm': algorithm,
                            'element_count': element_count,
                            'bytes_per_element': bytes_per_element,
                            'time_ns': time_ns,
                            'time_us': time_ns / 1000,
                            'time_ms': time_ns / 1_000_000,
                            'time_s': time_s,
                            'throughput_gib_s': throughput_gib_s,
                            'throughput_gb_s': throughput_gb_s,
                            'elements_per_second': elements_per_second,
                            'file_path': str(estimates_file)
                        }
                        
                        results.append(result)
                        
                    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                        print(f"Error processing {estimates_file}: {e}")
    
    return results


def export_to_csv(results: List[Dict], output_file: str):
    """Export results to CSV format."""
    if not results:
        return
    
    fieldnames = [
        'vector_size_name', 'size_display', 'size_bytes', 'size_kb', 'size_mb',
        'algorithm', 'element_count', 'bytes_per_element',
        'time_ns', 'time_us', 'time_ms', 'time_s',
        'throughput_gib_s', 'throughput_gb_s', 'elements_per_second',
        'file_path'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def export_to_json(results: List[Dict], output_file: str):
    """Export results to JSON format."""
    output_data = {
        'metadata': {
            'total_measurements': len(results),
            'algorithms': sorted(list(set(r['algorithm'] for r in results))),
            'vector_sizes': sorted(list(set(r['vector_size_name'] for r in results))),
            'size_range_bytes': {
                'min': min(r['size_bytes'] for r in results) if results else 0,
                'max': max(r['size_bytes'] for r in results) if results else 0
            }
        },
        'measurements': results
    }
    
    with open(output_file, 'w') as jsonfile:
        json.dump(output_data, jsonfile, indent=2)


def main():
    criterion_dir = "/home/mtantaoui/repos/mathlib/simdly/target/criterion"
    
    print("Extracting benchmark data for export...")
    results = extract_benchmark_data(criterion_dir)
    
    # Sort results by size then algorithm
    results.sort(key=lambda x: (x['size_bytes'], x['algorithm']))
    
    print(f"Found {len(results)} measurements")
    
    # Export to CSV
    csv_file = "/home/mtantaoui/repos/mathlib/simdly/benchmark_data.csv"
    export_to_csv(results, csv_file)
    print(f"Data exported to CSV: {csv_file}")
    
    # Export to JSON
    json_file = "/home/mtantaoui/repos/mathlib/simdly/benchmark_data.json"
    export_to_json(results, json_file)
    print(f"Data exported to JSON: {json_file}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total measurements: {len(results)}")
    
    algorithms = set(r['algorithm'] for r in results)
    print(f"Algorithms tested: {', '.join(sorted(algorithms))}")
    
    sizes = set(r['size_display'] for r in results)
    print(f"Vector sizes: {', '.join(sorted(sizes, key=lambda x: float(x.split()[0])))}")
    
    print(f"\nBest performance by algorithm:")
    by_algo = {}
    for result in results:
        algo = result['algorithm']
        if algo not in by_algo:
            by_algo[algo] = []
        by_algo[algo].append(result)
    
    for algo in sorted(by_algo.keys()):
        best = max(by_algo[algo], key=lambda x: x['throughput_gib_s'])
        print(f"  {algo:<12}: {best['throughput_gib_s']:.2f} GiB/s at {best['size_display']}")


if __name__ == "__main__":
    main()