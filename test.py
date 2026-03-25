import torch
import time

def test_pytorch_setup():
    print("-" * 30)
    print(f"1. PyTorch 版本: {torch.__version__}")
    
    # 检查 MPS (Apple Silicon 显卡加速)
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    
    print(f"2. Mac 硬件加速 (MPS) 是否可用: {mps_available}")
    print(f"3. PyTorch 是否支持 MPS 编译: {mps_built}")
    
    # 确定计算设备
    if mps_available:
        device = torch.device("mps")
        print(">>> 结论: 正在使用 Mac GPU (MPS) 加速！ ✅")
    else:
        device = torch.device("cpu")
        print(">>> 结论: 未检测到 GPU 加速，正在使用 CPU。 ⚠️")
    
    print("-" * 30)
    
    # 简单的运算压力测试
    print(f"正在 {device} 上进行矩阵运算测试...")
    
    # 创建两个大的随机矩阵
    size = 4000
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # 开始计时
    start_time = time.time()
    
    # 执行矩阵乘法
    z = torch.matmul(x, y)
    
    # 确保同步（MPS 需要同步才能准确计时）
    if mps_available:
        torch.mps.synchronize()
        
    end_time = time.time()
    
    print(f"计算完成！耗时: {end_time - start_time:.4f} 秒")
    print("-" * 30)

if __name__ == "__main__":
    try:
        test_pytorch_setup()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("提示: 这通常是因为 Cursor 选错了 Python 解释器，或者该环境下没装 torch。")