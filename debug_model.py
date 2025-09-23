import numpy as np
import torch
from scipy.fftpack import dct
from Model import UNet, UNet_CA, UNet_CA6, UNet_CA7

def debug_model_dimensions():
    # 创建模拟数据
    batch_size = 2
    spec_length = 2500
    
    # 模拟噪声数据
    noise = np.random.randn(batch_size, spec_length).astype(np.float32)  # 确保是float32
    print(f"Noise shape: {noise.shape}")
    
    spectra_num, spec = noise.shape
    print(f"spectra_num: {spectra_num}, spec: {spec}")
    
    # 模拟干净光谱生成
    clean_spectra = np.random.randn(spec, spectra_num).astype(np.float32)  # 注意这里的转置
    print(f"Clean spectra shape: {clean_spectra.shape}")
    
    # 转置
    clean_spectra = clean_spectra.T
    print(f"After transpose - Clean spectra shape: {clean_spectra.shape}")
    
    noisy_spectra = clean_spectra + noise
    print(f"Noisy spectra shape: {noisy_spectra.shape}")
    
    # 定义输入输出
    input_coef = np.zeros(np.shape(noisy_spectra), dtype=np.float32)
    output_coef = np.zeros(np.shape(noise), dtype=np.float32)
    print(f"input_coef shape: {input_coef.shape}")
    print(f"output_coef shape: {output_coef.shape}")
    
    # 进行预处理, dct变换
    for index in range(spectra_num):
        input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
        output_coef[index, :] = dct(noise[index, :], norm='ortho')
    
    # reshape 成3维度
    input_coef = np.reshape(input_coef, (-1, 1, spec))
    output_coef = np.reshape(output_coef, (-1, 1, spec))
    print(f"After reshape - input_coef shape: {input_coef.shape}")
    print(f"After reshape - output_coef shape: {output_coef.shape}")
    
    # 转换为tensor
    input_coef = torch.from_numpy(input_coef)
    output_coef = torch.from_numpy(output_coef)
    print(f"Tensor input_coef shape: {input_coef.shape}")
    print(f"Tensor output_coef shape: {output_coef.shape}")
    
    # 测试不同的模型
    models = {
        'UNet': UNet(1, 1),
        'UNet_CA': UNet_CA(1, 1),
        'UNet_CA6': UNet_CA6(1, 1),
        'UNet_CA7': UNet_CA7(1, 1)
    }
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        try:
            print(f"Model structure: {model}")
            # 确保模型参数也是float32类型
            model = model.float()
            preds = model(input_coef)
            print(f"Output shape: {preds.shape}")
            print(f"Expected output shape: {output_coef.shape}")
            print("SUCCESS")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    debug_model_dimensions()