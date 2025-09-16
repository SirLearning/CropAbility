#!/usr/bin/env python3
"""
PGL 主运行脚本
提供命令行接口来执行各种操作
"""

import argparse
import sys
import os

# 添加项目路径到Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='PGL - Performance GPU Library')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 导出模型命令
    export_parser = subparsers.add_parser('export', help='导出TorchScript模型')
    export_parser.add_argument('--output', '-o', default='triton_add_model.pt',
                             help='输出模型文件路径')
    export_parser.add_argument('--trace', action='store_true',
                             help='使用trace模式而不是script模式')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试和验证')
    test_parser.add_argument('--model', '-m', default='triton_add_model.pt',
                           help='要测试的模型文件路径')
    test_parser.add_argument('--correctness', action='store_true',
                           help='运行正确性验证')
    test_parser.add_argument('--benchmark', action='store_true',
                           help='运行性能基准测试')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    
    args = parser.parse_args()
    
    if args.command == 'export':
        from pgl.ops.export import export_torchscript_model
        try:
            model = export_torchscript_model(args.output, use_trace=args.trace)
            print(f"✓ 模型成功导出到: {args.output}")
        except Exception as e:
            print(f"✗ 导出失败: {e}")
            sys.exit(1)
            
    elif args.command == 'test':
        if args.correctness:
            from pgl.ops.test import validate_correctness
            print("=== 运行正确性验证 ===")
            success = validate_correctness()
            print(f"正确性验证: {'通过' if success else '失败'}")
            
        if args.benchmark:
            from pgl.ops.test import benchmark_add_operations
            print("=== 运行性能基准测试 ===")
            results = benchmark_add_operations()
            
            if results['sizes']:
                print("\n结果摘要:")
                for i, size in enumerate(results['sizes']):
                    print(f"大小 {size:>8}: "
                          f"Triton {results['triton_times'][i]:>6.3f}ms, "
                          f"PyTorch {results['pytorch_times'][i]:>6.3f}ms, "
                          f"加速比 {results['speedup_ratios'][i]:>5.2f}x")
        
        if hasattr(args, 'model') and not (args.correctness or args.benchmark):
            from pgl.ops.export import test_exported_model
            print(f"=== 测试模型: {args.model} ===")
            success = test_exported_model(args.model)
            print(f"模型测试: {'通过' if success else '失败'}")
            
    elif args.command == 'info':
        from pgl.ops.test import print_system_info
        print_system_info()
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
