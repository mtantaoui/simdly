/**
 * Parses Criterion benchmark data to extract performance metrics for charts
 * Reads JSON files from target/criterion and calculates speedup ratios
 */

import fs from 'fs';
import path from 'path';

// Path to criterion benchmark results (relative to project root)
const CRITERION_PATH = '../target/criterion';

/**
 * Reads estimate JSON data from criterion output
 * @param {string} benchmarkPath - Path to benchmark directory
 * @returns {Object|null} - Parsed estimates or null if not found
 */
function readEstimates(benchmarkPath) {
  try {
    const estimatesPath = path.join(benchmarkPath, 'new', 'estimates.json');
    if (!fs.existsSync(estimatesPath)) {
      console.warn(`Estimates not found: ${estimatesPath}`);
      return null;
    }
    
    const data = fs.readFileSync(estimatesPath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error(`Error reading estimates from ${benchmarkPath}:`, error);
    return null;
  }
}

/**
 * Extracts performance time from estimates (in nanoseconds)
 * @param {Object} estimates - Criterion estimates object
 * @returns {number} - Mean time in nanoseconds
 */
function extractTime(estimates) {
  if (!estimates || !estimates.mean) {
    return null;
  }
  return estimates.mean.point_estimate;
}

/**
 * Calculates speedup ratio between baseline and optimized implementation
 * @param {number} baselineTime - Scalar baseline time
 * @param {number} optimizedTime - SIMD optimized time  
 * @returns {number} - Speedup ratio (e.g., 3.2x)
 */
function calculateSpeedup(baselineTime, optimizedTime) {
  if (!baselineTime || !optimizedTime || optimizedTime === 0) {
    return 1.0;
  }
  return baselineTime / optimizedTime;
}

/**
 * Parses benchmark data for a specific category and data size
 * @param {string} category - Benchmark category (e.g., 'Basic_Operations')
 * @param {string} dataSize - Data size (e.g., 'Medium_64KB')
 * @returns {Object} - Parsed benchmark data with speedups
 */
function parseBenchmarkGroup(category, dataSize) {
  const groupPath = path.join(CRITERION_PATH, `${category}_${dataSize}`);
  
  if (!fs.existsSync(groupPath)) {
    console.warn(`Benchmark group not found: ${groupPath}`);
    return null;
  }

  const results = {};
  
  // Read all benchmark directories in this group
  const benchmarks = fs.readdirSync(groupPath)
    .filter(dir => fs.statSync(path.join(groupPath, dir)).isDirectory())
    .filter(dir => dir !== 'report'); // Skip report directory

  // Group benchmarks by function name
  const functionGroups = {};
  
  benchmarks.forEach(benchmark => {
    // Extract function name and implementation type
    // Format: {function}_{implementation}_{size?}
    const parts = benchmark.split('_');
    const functionName = parts[0];
    const implementation = parts.slice(1).join('_');
    
    if (!functionGroups[functionName]) {
      functionGroups[functionName] = {};
    }
    
    const estimates = readEstimates(path.join(groupPath, benchmark));
    if (estimates) {
      const time = extractTime(estimates);
      if (time) {
        functionGroups[functionName][implementation] = time;
      }
    }
  });

  // Calculate speedups for each function
  Object.keys(functionGroups).forEach(functionName => {
    const group = functionGroups[functionName];
    
    // Find baseline (scalar) time
    const baselineKey = Object.keys(group).find(key => 
      key.includes('Scalar') || key.includes('Baseline')
    );
    
    if (!baselineKey || !group[baselineKey]) {
      console.warn(`No baseline found for ${functionName}`);
      return;
    }
    
    const baselineTime = group[baselineKey];
    
    results[functionName] = {
      baseline: baselineTime,
      implementations: {}
    };
    
    // Calculate speedup for each implementation
    Object.keys(group).forEach(implKey => {
      const time = group[implKey];
      const speedup = calculateSpeedup(baselineTime, time);
      
      // Categorize implementation type
      let type = 'other';
      if (implKey.includes('Scalar') || implKey.includes('Baseline')) {
        type = 'scalar';
      } else if (implKey.includes('Parallel')) {
        type = 'parallel';
      } else if (implKey.includes('SIMD')) {
        type = 'simd';
      }
      
      results[functionName].implementations[type] = {
        time,
        speedup
      };
    });
  });

  return results;
}

/**
 * Generates chart data structure from parsed benchmark results
 * @param {Object} benchmarkResults - Results from parseBenchmarkGroup
 * @returns {Object} - Chart.js compatible data structure
 */
function generateChartData(benchmarkResults) {
  if (!benchmarkResults) {
    return null;
  }

  const functions = Object.keys(benchmarkResults);
  const chartData = {
    functions,
    scalar: [],
    simd: [],
    parallel: []
  };

  functions.forEach(functionName => {
    const result = benchmarkResults[functionName];
    
    chartData.scalar.push(1.0); // Always 1x baseline
    
    const simdSpeedup = result.implementations.simd?.speedup || 1.0;
    const parallelSpeedup = result.implementations.parallel?.speedup || 1.0;
    
    chartData.simd.push(Number(simdSpeedup.toFixed(1)));
    chartData.parallel.push(Number(parallelSpeedup.toFixed(1)));
  });

  return chartData;
}

/**
 * Main function to extract all performance data for landing page charts
 * @returns {Object} - Complete performance data for all categories
 */
export function extractPerformanceData() {
  const categories = [
    { key: 'basic', name: 'Basic_Operations' },
    { key: 'trig', name: 'Trigonometric' },
    { key: 'exp', name: 'Exponential_Logarithmic' }
  ];
  
  const dataSize = 'Medium_64KB'; // Use medium size for landing page
  const performanceData = {};

  categories.forEach(({ key, name }) => {
    console.log(`Parsing ${name} benchmarks...`);
    
    const benchmarkResults = parseBenchmarkGroup(name, dataSize);
    const chartData = generateChartData(benchmarkResults);
    
    if (chartData) {
      performanceData[key] = {
        title: `${name.replace('_', ' ')} Performance`,
        ...chartData
      };
      
      console.log(`✅ ${name}: ${chartData.functions.length} functions parsed`);
    } else {
      console.warn(`❌ Failed to parse ${name} benchmarks`);
      
      // Provide fallback data
      performanceData[key] = {
        title: `${name.replace('_', ' ')} Performance (Sample Data)`,
        functions: ['sample'],
        scalar: [1.0],
        simd: [3.5],
        parallel: [5.2]
      };
    }
  });

  return performanceData;
}

/**
 * Development helper: prints summary of available benchmark data
 */
export function printBenchmarkSummary() {
  if (!fs.existsSync(CRITERION_PATH)) {
    console.log('❌ No criterion benchmarks found. Run: cargo bench --bench simd_comprehensive');
    return;
  }

  console.log('📊 Available Criterion Benchmarks:');
  
  const categories = fs.readdirSync(CRITERION_PATH)
    .filter(dir => fs.statSync(path.join(CRITERION_PATH, dir)).isDirectory())
    .filter(dir => dir !== 'report' && dir !== 'tmp');
  
  categories.forEach(category => {
    const categoryPath = path.join(CRITERION_PATH, category);
    const benchmarks = fs.readdirSync(categoryPath)
      .filter(dir => fs.statSync(path.join(categoryPath, dir)).isDirectory())
      .filter(dir => dir !== 'report');
      
    console.log(`  ${category}: ${benchmarks.length} benchmarks`);
  });
}

// Development mode: print summary if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  printBenchmarkSummary();
  const data = extractPerformanceData();
  console.log('\n📈 Extracted Performance Data:', JSON.stringify(data, null, 2));
}