#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型修复脚本 - 自动删除不兼容的模型文件并重新训练
"""

import os
import subprocess
import sys

def main():
    print("🔧 比特币预测系统 - 模型修复工具")
    print("=" * 50)
    
    # 检查是否存在模型文件
    if os.path.exists('model.pth'):
        print("🔍 发现旧模型文件: model.pth")
        
        # 询问用户是否删除
        response = input("是否删除旧模型并重新训练？(y/n): ").lower()
        if response in ['y', 'yes', '是']:
            try:
                # 删除旧模型文件
                os.remove('model.pth')
                print("✅ 已删除旧模型文件")
                
                # 开始重新训练
                print("🚀 开始重新训练模型...")
                print("💡 这个过程可能需要几分钟时间，请耐心等待")
                print("=" * 50)
                
                # 运行训练命令
                cmd = [sys.executable, 'bitcoin_prediction.py', '--train']
                result = subprocess.run(cmd)
                
                if result.returncode == 0:
                    print("=" * 50)
                    print("🎉 模型训练完成！")
                    print("💡 现在可以使用以下命令启动系统:")
                    print("   python start_continuous.py")
                    print("   或者")
                    print("   python bitcoin_prediction.py --continuous")
                else:
                    print("❌ 训练过程中出现错误")
                    
            except Exception as e:
                print(f"❌ 删除模型文件失败: {e}")
        else:
            print("👋 用户取消操作")
    else:
        print("ℹ️ 没有发现模型文件")
        print("🚀 开始训练新模型...")
        
        try:
            # 运行训练命令
            cmd = [sys.executable, 'bitcoin_prediction.py', '--train']
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                print("🎉 模型训练完成！")
            else:
                print("❌ 训练过程中出现错误")
        except Exception as e:
            print(f"❌ 训练失败: {e}")

if __name__ == "__main__":
    main() 