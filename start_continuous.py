#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比特币价格预测系统 - 快速启动脚本
自动启用连续实时监控模式（默认模式）
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动比特币价格预测系统")
    print("📊 连续实时分析模式（默认启动模式）")
    print("⚡ 智能触发：价格变化≥0.05%时进行分析")
    print("🎯 极高置信度(≥75%)时提示交易机会")
    print("🎰 事件合约模式：投注5u，预测正确获得9u")
    print("⌨️  运行中按键：'s' - 查看统计  'q' - 退出")
    print("=" * 50)
    
    # 检查主程序是否存在
    if not os.path.exists('bitcoin_prediction.py'):
        print("❌ 找不到 bitcoin_prediction.py 文件")
        return
    
    try:
        # 直接启动，默认就是连续分析模式
        cmd = [sys.executable, 'bitcoin_prediction.py']
        
        print("🔄 正在启动预测系统...")
        print("💡 提示：按 Ctrl+C 可停止程序")
        print("=" * 50)
        
        # 运行主程序
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 用户中断程序")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main() 