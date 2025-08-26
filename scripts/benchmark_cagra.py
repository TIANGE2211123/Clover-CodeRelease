#!/usr/bin/env python3
"""
CAGRA性能测试脚本

用于比较CAGRA算法与其他kNN算法的性能差异
"""

import subprocess
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

class CagraBenchmark:
    def __init__(self, executable_path="./build/linear-scans"):
        self.executable_path = executable_path
        self.results = {}
        
    def run_algorithm(self, algorithm_id, dataset_size, k_value, repetitions=3):
        """运行指定算法并测量性能"""
        print(f"Running algorithm {algorithm_id} with dataset size {dataset_size}, k={k_value}")
        
        times = []
        for i in range(repetitions):
            start_time = time.time()
            
            # 运行算法
            cmd = [self.executable_path, str(algorithm_id)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Error running algorithm {algorithm_id}: {result.stderr}")
                    return None
                    
                # 解析输出获取时间
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if f"({dataset_size}, {k_value}" in line:
                        # 提取时间数据
                        time_data = line.split('[')[1].split(']')[0]
                        times_str = time_data.split(',')
                        times.extend([float(t.strip()) for t in times_str if t.strip()])
                        break
                        
            except subprocess.TimeoutExpired:
                print(f"Algorithm {algorithm_id} timed out")
                return None
            except Exception as e:
                print(f"Error running algorithm {algorithm_id}: {e}")
                return None
                
        if times:
            return {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'times': times
            }
        return None
        
    def benchmark_algorithms(self, dataset_sizes, k_values, algorithms):
        """对多个算法进行性能测试"""
        results = {}
        
        for dataset_size in dataset_sizes:
            results[dataset_size] = {}
            for k in k_values:
                results[dataset_size][k] = {}
                for alg_name, alg_id in algorithms.items():
                    print(f"\nBenchmarking {alg_name} with dataset size {dataset_size}, k={k}")
                    result = self.run_algorithm(alg_id, dataset_size, k)
                    if result:
                        results[dataset_size][k][alg_name] = result
                    else:
                        print(f"Failed to get results for {alg_name}")
                        
        return results
        
    def save_results(self, results, output_file="cagra_benchmark_results.json"):
        """保存测试结果到JSON文件"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
    def load_results(self, input_file="cagra_benchmark_results.json"):
        """从JSON文件加载测试结果"""
        with open(input_file, 'r') as f:
            return json.load(f)
            
    def plot_performance_comparison(self, results, output_dir="plots"):
        """绘制性能对比图"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 提取算法名称
        algorithms = set()
        for dataset_size in results:
            for k in results[dataset_size]:
                algorithms.update(results[dataset_size][k].keys())
        algorithms = sorted(list(algorithms))
        
        # 为每个k值创建图表
        for k in set().union(*[set(results[ds].keys()) for ds in results]):
            plt.figure(figsize=(12, 8))
            
            dataset_sizes = sorted(results.keys())
            for alg in algorithms:
                times = []
                for ds in dataset_sizes:
                    if k in results[ds] and alg in results[ds][k]:
                        times.append(results[ds][k][alg]['mean_time'])
                    else:
                        times.append(None)
                        
                # 过滤掉None值
                valid_sizes = [ds for ds, t in zip(dataset_sizes, times) if t is not None]
                valid_times = [t for t in times if t is not None]
                
                if valid_times:
                    plt.plot(valid_sizes, valid_times, marker='o', label=alg, linewidth=2)
                    
            plt.xlabel('Dataset Size')
            plt.ylabel('Execution Time (ns)')
            plt.title(f'Performance Comparison for k={k}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.xscale('log')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_k{k}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_speedup_analysis(self, results, baseline_alg="bitonic", output_dir="plots"):
        """绘制相对于基准算法的加速比分析"""
        Path(output_dir).mkdir(exist_ok=True)
        
        algorithms = set()
        for dataset_size in results:
            for k in results[dataset_size]:
                algorithms.update(results[dataset_size][k].keys())
        algorithms = sorted(list(algorithms))
        
        if baseline_alg not in algorithms:
            print(f"Baseline algorithm {baseline_alg} not found in results")
            return
            
        # 为每个k值创建加速比图表
        for k in set().union(*[set(results[ds].keys()) for ds in results]):
            plt.figure(figsize=(12, 8))
            
            dataset_sizes = sorted(results.keys())
            for alg in algorithms:
                if alg == baseline_alg:
                    continue
                    
                speedups = []
                for ds in dataset_sizes:
                    if (k in results[ds] and alg in results[ds][k] and 
                        baseline_alg in results[ds][k]):
                        baseline_time = results[ds][k][baseline_alg]['mean_time']
                        alg_time = results[ds][k][alg]['mean_time']
                        speedup = baseline_time / alg_time
                        speedups.append(speedup)
                    else:
                        speedups.append(None)
                        
                # 过滤掉None值
                valid_sizes = [ds for ds, s in zip(dataset_sizes, speedups) if s is not None]
                valid_speedups = [s for s in speedups if s is not None]
                
                if valid_speedups:
                    plt.plot(valid_sizes, valid_speedups, marker='o', label=alg, linewidth=2)
                    
            plt.xlabel('Dataset Size')
            plt.ylabel(f'Speedup vs {baseline_alg}')
            plt.title(f'Speedup Analysis for k={k}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/speedup_k{k}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    def generate_report(self, results, output_file="cagra_benchmark_report.md"):
        """生成性能测试报告"""
        with open(output_file, 'w') as f:
            f.write("# CAGRA性能测试报告\n\n")
            
            # 算法列表
            algorithms = set()
            for dataset_size in results:
                for k in results[dataset_size]:
                    algorithms.update(results[dataset_size][k].keys())
            algorithms = sorted(list(algorithms))
            
            f.write("## 测试算法\n\n")
            for alg in algorithms:
                f.write(f"- {alg}\n")
            f.write("\n")
            
            # 数据集大小
            dataset_sizes = sorted(results.keys())
            f.write(f"## 数据集大小\n\n")
            f.write(f"测试的数据集大小: {', '.join(map(str, dataset_sizes))}\n\n")
            
            # 性能表格
            for k in set().union(*[set(results[ds].keys()) for ds in results]):
                f.write(f"## k={k} 性能结果\n\n")
                f.write("| 算法 | 数据集大小 | 平均时间 (ns) | 标准差 | 最小时间 | 最大时间 |\n")
                f.write("|------|------------|---------------|--------|----------|----------|\n")
                
                for ds in dataset_sizes:
                    if k in results[ds]:
                        for alg in algorithms:
                            if alg in results[ds][k]:
                                result = results[ds][k][alg]
                                f.write(f"| {alg} | {ds} | {result['mean_time']:.2f} | "
                                       f"{result['std_time']:.2f} | {result['min_time']:.2f} | "
                                       f"{result['max_time']:.2f} |\n")
                f.write("\n")
                
            # 加速比分析
            baseline_alg = "bitonic"
            if baseline_alg in algorithms:
                f.write(f"## 相对于 {baseline_alg} 的加速比\n\n")
                f.write("| 算法 | 数据集大小 | k值 | 加速比 |\n")
                f.write("|------|------------|-----|--------|\n")
                
                for ds in dataset_sizes:
                    for k in results[ds]:
                        if baseline_alg in results[ds][k]:
                            baseline_time = results[ds][k][baseline_alg]['mean_time']
                            for alg in algorithms:
                                if alg != baseline_alg and alg in results[ds][k]:
                                    alg_time = results[ds][k][alg]['mean_time']
                                    speedup = baseline_time / alg_time
                                    f.write(f"| {alg} | {ds} | {k} | {speedup:.2f}x |\n")
                f.write("\n")
                
        print(f"Report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="CAGRA性能测试脚本")
    parser.add_argument("--executable", default="./build/linear-scans", 
                       help="可执行文件路径")
    parser.add_argument("--dataset-sizes", nargs="+", type=int, 
                       default=[1000, 5000, 10000, 50000, 100000],
                       help="测试的数据集大小")
    parser.add_argument("--k-values", nargs="+", type=int, 
                       default=[32, 64, 128],
                       help="测试的k值")
    parser.add_argument("--algorithms", nargs="+", 
                       default=["bitonic", "warpwise", "hubs", "cagra"],
                       help="要测试的算法")
    parser.add_argument("--load-results", help="从文件加载结果")
    parser.add_argument("--save-results", default="cagra_benchmark_results.json",
                       help="保存结果的文件名")
    parser.add_argument("--generate-plots", action="store_true",
                       help="生成性能图表")
    parser.add_argument("--generate-report", action="store_true",
                       help="生成性能报告")
    
    args = parser.parse_args()
    
    # 算法ID映射
    algorithm_ids = {
        "bitonic": 0,
        "warpwise": 1,
        "hubs": 2,
        "hubs_ws": 3,
        "faiss": 4,
        "faiss_ws": 5,
        "faiss_bs": 6,
        "treelogy_kdtree": 7,
        "cagra": 8
    }
    
    # 过滤有效的算法
    valid_algorithms = {name: algorithm_ids[name] for name in args.algorithms 
                       if name in algorithm_ids}
    
    if not valid_algorithms:
        print("No valid algorithms specified")
        sys.exit(1)
        
    benchmark = CagraBenchmark(args.executable)
    
    if args.load_results:
        print(f"Loading results from {args.load_results}")
        results = benchmark.load_results(args.load_results)
    else:
        print("Running performance benchmarks...")
        results = benchmark.benchmark_algorithms(
            args.dataset_sizes, args.k_values, valid_algorithms
        )
        
        if args.save_results:
            benchmark.save_results(results, args.save_results)
    
    if args.generate_plots:
        print("Generating performance plots...")
        benchmark.plot_performance_comparison(results)
        benchmark.plot_speedup_analysis(results)
        
    if args.generate_report:
        print("Generating performance report...")
        benchmark.generate_report(results)
        
    print("Benchmark completed!")

if __name__ == "__main__":
    main()
