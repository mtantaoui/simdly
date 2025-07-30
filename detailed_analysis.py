#!/usr/bin/env python3
"""
Detailed analysis of the benchmark results with performance insights
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
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
                        
                        result = {
                            'vector_size_name': addition_dir.name,
                            'size_bytes': size_bytes,
                            'size_kb': size_bytes / 1024,
                            'algorithm': algorithm,
                            'element_count': element_count,
                            'time_ns': time_ns,
                            'throughput_gib_s': throughput_gib_s,
                            'bytes_per_element': size_bytes / element_count if element_count > 0 else 0
                        }
                        
                        results.append(result)
                        
                    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                        print(f"Error processing {estimates_file}: {e}")
    
    return results


def analyze_performance_patterns(results: List[Dict]) -> str:
    """Analyze performance patterns and generate insights."""
    if not results:
        return "No data to analyze."
    
    output = []
    output.append("=== DETAILED PERFORMANCE ANALYSIS ===\n")
    
    # 1. Performance by data size analysis
    output.append("1. PERFORMANCE SCALING BY DATA SIZE:")
    output.append("-" * 50)
    
    # Group by algorithm to see scaling patterns
    by_algorithm = {}
    for result in results:
        algo = result['algorithm']
        if algo not in by_algorithm:
            by_algorithm[algo] = []
        by_algorithm[algo].append(result)
    
    for algorithm in sorted(by_algorithm.keys()):
        algo_results = sorted(by_algorithm[algorithm], key=lambda x: x['size_bytes'])
        output.append(f"\n{algorithm.upper()} Scaling:")
        
        for result in algo_results:
            size_kb = result['size_kb']
            throughput = result['throughput_gib_s']
            output.append(f"  {size_kb:>8.1f} KB -> {throughput:>6.2f} GiB/s")
    
    # 2. Algorithm comparison at each size
    output.append(f"\n\n2. ALGORITHM COMPARISON BY SIZE:")
    output.append("-" * 50)
    
    # Group by size
    by_size = {}
    for result in results:
        size = result['vector_size_name']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(result)
    
    # Sort sizes by bytes
    sorted_sizes = sorted(by_size.keys(), 
                         key=lambda x: by_size[x][0]['size_bytes'])
    
    for size in sorted_sizes:
        size_results = sorted(by_size[size], 
                            key=lambda x: -x['throughput_gib_s'])  # Sort by throughput desc
        
        size_kb = size_results[0]['size_kb']
        output.append(f"\n{size} ({size_kb:.1f} KB):")
        
        # Find best performer
        best_throughput = max(r['throughput_gib_s'] for r in size_results)
        
        for result in size_results:
            algo = result['algorithm']
            throughput = result['throughput_gib_s']
            ratio = throughput / best_throughput if best_throughput > 0 else 0
            performance_indicator = "🏆" if throughput == best_throughput else ""
            
            output.append(f"  {algo:<12} {throughput:>6.2f} GiB/s ({ratio:>5.1%}) {performance_indicator}")
    
    # 3. Sweet spots and bottlenecks
    output.append(f"\n\n3. PERFORMANCE INSIGHTS:")
    output.append("-" * 50)
    
    # Find peak performance for each algorithm
    peak_performances = {}
    for algo in by_algorithm:
        algo_results = by_algorithm[algo]
        peak = max(algo_results, key=lambda x: x['throughput_gib_s'])
        peak_performances[algo] = peak
    
    output.append("\nPeak Performance by Algorithm:")
    for algo, peak in sorted(peak_performances.items(), 
                           key=lambda x: -x[1]['throughput_gib_s']):
        size_kb = peak['size_kb']
        throughput = peak['throughput_gib_s']
        output.append(f"  {algo:<12} {throughput:>6.2f} GiB/s at {size_kb:.1f} KB")
    
    # Find where SIMD starts to beat scalar
    output.append("\nSIMD vs Scalar Analysis:")
    simd_results = {r['size_bytes']: r for r in by_algorithm.get('simd', [])}
    scalar_results = {r['size_bytes']: r for r in by_algorithm.get('scalar', [])}
    
    common_sizes = set(simd_results.keys()) & set(scalar_results.keys())
    
    simd_wins = []
    scalar_wins = []
    
    for size_bytes in sorted(common_sizes):
        simd_perf = simd_results[size_bytes]['throughput_gib_s']
        scalar_perf = scalar_results[size_bytes]['throughput_gib_s']
        size_kb = size_bytes / 1024
        
        if simd_perf > scalar_perf:
            simd_wins.append((size_kb, simd_perf, scalar_perf))
        else:
            scalar_wins.append((size_kb, scalar_perf, simd_perf))
    
    if simd_wins:
        output.append("  SIMD outperforms Scalar at:")
        for size_kb, simd_perf, scalar_perf in simd_wins:
            improvement = (simd_perf - scalar_perf) / scalar_perf * 100
            output.append(f"    {size_kb:>8.1f} KB: {simd_perf:.2f} vs {scalar_perf:.2f} GiB/s (+{improvement:.1f}%)")
    
    if scalar_wins:
        output.append("  Scalar outperforms SIMD at:")
        for size_kb, scalar_perf, simd_perf in scalar_wins:
            advantage = (scalar_perf - simd_perf) / simd_perf * 100
            output.append(f"    {size_kb:>8.1f} KB: {scalar_perf:.2f} vs {simd_perf:.2f} GiB/s (+{advantage:.1f}%)")
    
    # Parallel SIMD analysis
    if 'parallel_simd' in by_algorithm:
        output.append("\nParallel SIMD Analysis:")
        parallel_results = {r['size_bytes']: r for r in by_algorithm['parallel_simd']}
        
        # Compare with regular SIMD where both exist
        common_parallel_sizes = set(parallel_results.keys()) & set(simd_results.keys())
        
        for size_bytes in sorted(common_parallel_sizes):
            parallel_perf = parallel_results[size_bytes]['throughput_gib_s']
            simd_perf = simd_results[size_bytes]['throughput_gib_s']
            size_kb = size_bytes / 1024
            
            if parallel_perf > simd_perf:
                improvement = (parallel_perf - simd_perf) / simd_perf * 100
                output.append(f"  {size_kb:>8.1f} KB: Parallel SIMD wins {parallel_perf:.2f} vs {simd_perf:.2f} (+{improvement:.1f}%)")
            else:
                penalty = (simd_perf - parallel_perf) / simd_perf * 100
                output.append(f"  {size_kb:>8.1f} KB: Regular SIMD wins {simd_perf:.2f} vs {parallel_perf:.2f} (+{penalty:.1f}%)")
    
    # 4. Raw data table
    output.append(f"\n\n4. COMPLETE DATA TABLE:")
    output.append("-" * 80)
    output.append(f"{'Size (KB)':<10} {'Algorithm':<12} {'Elements':<10} {'Time (ns)':<12} {'Throughput':<12}")
    output.append("-" * 80)
    
    # Sort all results by size then by throughput
    all_results = sorted(results, key=lambda x: (x['size_bytes'], -x['throughput_gib_s']))
    
    for result in all_results:
        size_kb = result['size_kb']
        algo = result['algorithm']
        elements = result['element_count']
        time_ns = result['time_ns']
        throughput = result['throughput_gib_s']
        
        output.append(f"{size_kb:<10.1f} {algo:<12} {elements:<10} {time_ns:<12.2f} {throughput:<12.2f}")
    
    return "\n".join(output)


def main():
    criterion_dir = "/home/mtantaoui/repos/mathlib/simdly/target/criterion"
    
    print("Extracting benchmark data for detailed analysis...")
    results = extract_benchmark_data(criterion_dir)
    
    print(f"Analyzing {len(results)} benchmark measurements...")
    
    # Generate detailed analysis
    analysis = analyze_performance_patterns(results)
    print("\n" + analysis)
    
    # Save to file
    output_file = "/home/mtantaoui/repos/mathlib/simdly/detailed_benchmark_analysis.txt"
    with open(output_file, 'w') as f:
        f.write(analysis)
    
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()