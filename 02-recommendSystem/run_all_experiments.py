"""
综合实验脚本：一键运行所有实验
包括：
1. 不同嵌入维度的实验
2. 不同邻域采样大小的实验
3. 不同聚合器类型的实验
"""
import subprocess
import sys
import time

def run_experiment(script_name, description):
    """运行单个实验脚本"""
    print("\n" + "=" * 70)
    print(f"开始运行实验: {description}")
    print("=" * 70)
    start_time = time.time()
    
    try:
        # 运行实验脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        elapsed_time = time.time() - start_time
        print(f"\n✓ 实验完成! 耗时: {elapsed_time:.2f}秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 实验失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 运行错误: {e}")
        return False

def main():
    experiments = [
        ("experiment_dim.py", "测试不同嵌入维度对AUC的影响"),
        ("experiment_neighbor.py", "测试不同邻域采样大小对AUC的影响"),
        ("experiment_aggregator.py", "测试不同聚合器类型对AUC的影响"),
    ]
    
    print("=" * 70)
    print(" " * 15 + "KGCN 推荐系统 - 综合实验")
    print("=" * 70)
    print(f"\n即将运行 {len(experiments)} 个实验...")
    print("\n实验列表:")
    for i, (script, desc) in enumerate(experiments, 1):
        print(f"  {i}. {desc}")
    
    input("\n按 Enter 键开始运行实验...")
    
    results = {}
    total_start_time = time.time()
    
    for script, desc in experiments:
        success = run_experiment(script, desc)
        results[desc] = "成功" if success else "失败"
        if not success:
            print(f"\n警告: 实验 '{desc}' 失败，是否继续？")
            choice = input("继续运行下一个实验? (Y/n): ")
            if choice.lower() == 'n':
                break
    
    total_elapsed_time = time.time() - total_start_time
    
    # 打印总结
    print("\n" + "=" * 70)
    print(" " * 25 + "实验总结")
    print("=" * 70)
    for desc, status in results.items():
        status_symbol = "✓" if status == "成功" else "✗"
        print(f"{status_symbol} {desc}: {status}")
    print(f"\n总耗时: {total_elapsed_time:.2f}秒 ({total_elapsed_time/60:.2f}分钟)")
    print("\n所有实验结果图片已保存在当前目录:")
    print("  - experiment_dim_results.png")
    print("  - experiment_neighbor_results.png")
    print("  - experiment_aggregator_results.png")
    print("=" * 70)

if __name__ == '__main__':
    main()

