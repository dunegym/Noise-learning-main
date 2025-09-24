import os
import sys
import shutil
from config import DefaultConfig

def test_modified_batch_predict():
    """测试修改后的Main.py中的batch_predict函数"""
    print("Testing modified batch_predict function...")
    
    # 添加当前目录到Python路径
    sys.path.append('.')
    
    # 导入Main.py中的函数
    import Main
    
    # 创建配置对象
    config = DefaultConfig()
    config.is_batch_predicting = True
    config.global_step = 5200
    config.predict_root = 'Predict'
    config.test_model_dir = 'Model'
    
    # 创建输出目录（如果不存在）
    result_dir = os.path.join('Result', 'Nanophoton', 'UNet_CA', 'step_5200')
    os.makedirs(result_dir, exist_ok=True)
    
    # 运行批处理预测
    try:
        Main.batch_predict(config)
        print("Batch prediction completed successfully!")
        
        # 检查输出文件
        output_files = os.listdir(result_dir)
        mat_files = [f for f in output_files if f.endswith('.mat')]
        
        if mat_files:
            print(f"Generated {len(mat_files)} output files:")
            for f in mat_files:
                print(f"  - {f}")
        else:
            print("No output files generated!")
            
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_modified_batch_predict()