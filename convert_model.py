#!/usr/bin/env python3
"""转换 YOLOv8 模型为 ONNX 格式"""

from ultralytics import YOLO
import os

def convert_model(model_name='yolov8n'):
    """下载并转换 YOLOv8 模型"""
    print(f"正在下载 {model_name} 模型...")
    model = YOLO(f'{model_name}.pt')
    
    print("正在转换为 ONNX 格式...")
    # 导出为 ONNX，使用640x640输入，简化模型
    model.export(
        format='onnx',
        imgsz=640,
        half=False,  # 不使用半精度，确保兼容性
        simplify=True,  # 简化模型
        opset=12  # ONNX opset版本
    )
    
    # 移动模型到 models 文件夹
    onnx_file = f'{model_name}.onnx'
    if os.path.exists(onnx_file):
        target_path = os.path.join('models', onnx_file)
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(onnx_file, target_path)
        print(f"✅ 转换完成！模型已保存到: {os.path.abspath(target_path)}")
        print(f"📦 文件大小: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
    else:
        print("❌ 转换失败，未找到输出文件")

if __name__ == '__main__':
    # 转换 yolov8n (轻量级，速度快)
    convert_model('yolov8n')
    
    # 如需更高精度，可以取消下面注释转换其他版本
    # convert_model('yolov8s')  # 小模型，精度更高
    # convert_model('yolov8m')  # 中等模型
