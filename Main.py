"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-13
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Model import UNet_CA, UNet, UNet_CA6, UNet_CA7
from Spectra_generator import spectra_generator
from Make_dataset import Read_data, Make_dataset
from config import DefaultConfig
import numpy as np
import os
from scipy.fftpack import dct, idct
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import glob
import h5py


# define the checkdir
def check_dir(config):
    if not os.path.exists(config.checkpoint):
        os.mkdir(config.checkpoint)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.test_dir):
        os.mkdir(config.test_dir)


# 定义测试结果保存路径
def test_result_dir(config):
    result_dir = os.path.join(config.test_dir, config.Instrument,
                              config.model_name, "step_" + str(config.global_step))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


# 定义模型保存路径
def save_model_dir(config):
    save_model_dir = os.path.join(config.checkpoint, config.Instrument,
                                  config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    return save_model_dir


# 定义日志路径
def save_log_dir(config):
    save_log_dir = os.path.join(config.logs, config.Instrument,
                                config.model_name, "batch_" + str(config.batch_size))
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    return save_log_dir


# 查找最新的checkpoint文件
def find_latest_checkpoint(save_model_path):
    checkpoint_files = glob.glob(os.path.join(save_model_path, "*.pt"))
    if not checkpoint_files:
        return None, 0
    
    # 提取步数并找到最大的
    steps = []
    for f in checkpoint_files:
        try:
            step = int(os.path.basename(f).split('.')[0])
            steps.append((step, f))
        except ValueError:
            continue
    
    if not steps:
        return None, 0
    
    # 返回步数最大的checkpoint
    steps.sort(key=lambda x: x[0], reverse=True)
    return steps[0][1], steps[0][0]


def train(config):
    # 如需指定可见 GPU，先设置环境变量（必须在 torch.cuda.is_available() 之前）
    if config.use_gpu and getattr(config, 'visible_devices', None):
        os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices  # e.g. '0,1'
    device = torch.device("cuda:0" if (config.use_gpu and torch.cuda.is_available()) else "cpu")
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    print(model)
    # 使用GPU
    if config.use_gpu and torch.cuda.is_available():
        # 多GPU训练
        model = nn.DataParallel(model)
        model = model.cuda()
        model.to(device)
    else:
        model.to(device)
    # 定义损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 定义学习速率调整
    schedual = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)
    # 保存模型的目录
    save_model_path = save_model_dir(config)
    # 保存日志的目录
    save_log = save_log_dir(config)
    # 全局训练步数
    global_step = 0
    
    # 根据配置决定是否加载预训练模型或最新的checkpoint
    if config.is_pretrain or config.resume_from_latest:
        if config.is_pretrain:
            # 加载指定的预训练模型
            global_step = config.global_step
            model_file = os.path.join(save_model_path, str(global_step) + '.pt')
            if not os.path.exists(model_file):
                # 如果指定的模型不存在，尝试从测试模型目录加载
                model_file = os.path.join(config.test_model_dir, str(global_step) + '.pt')
        else:
            # 查找最新的checkpoint文件
            model_file, global_step = find_latest_checkpoint(save_model_path)
            
        # 如果找到了模型文件，则加载
        if model_file and os.path.exists(model_file):
            # 加载模型参数
            if config.use_gpu and torch.cuda.is_available():
                kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
                state = torch.load(model_file, **kwargs)
            else:
                kwargs = {'map_location': torch.device('cpu')}
                state = torch.load(model_file, **kwargs)
            model.load_state_dict(state['model'])
            # 恢复优化器状态
            optimizer.load_state_dict(state['optimizer'])
            print('Successfully loaded the model saved at global step = {}'.format(global_step))
        else:
            if config.is_pretrain:
                print('Warning: Pretrained model file not found, starting from scratch')
            elif config.resume_from_latest:
                print('No checkpoint found, starting from scratch')
    elif config.is_pretrain:
        # 保持原有的预训练逻辑以确保向后兼容
        global_step = config.global_step
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        # 加载模型参数
        if config.use_gpu and torch.cuda.is_available():
            kwargs = {'map_location': lambda storage, loc: storage.cuda([0, 1])}
            state = torch.load(model_file, **kwargs)
        else:
            kwargs = {'map_location': torch.device('cpu')}
            state = torch.load(model_file, **kwargs)
        model.load_state_dict(state['model'])
        # 恢复优化器状态
        optimizer.load_state_dict(state['optimizer'])
        print('Successfully loaded the pretrained model saved at global step = {}'.format(global_step))
    
    # 加载数据集
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    # 制作Loader
    TrainSet, ValidSet = Make_dataset(train_set), Make_dataset(valid_set)

    train_loader = DataLoader(dataset=TrainSet, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=ValidSet, batch_size=256, pin_memory=True)
    # 仿真光谱生成器
    gen_train = spectra_generator()
    gen_valid = spectra_generator()
    # 定义tensorboard
    writer = SummaryWriter(save_log)
    for epoch in range(config.max_epoch):
        print(f'Epoch {epoch+1}/{config.max_epoch} starting...')
        epoch_loss = 0.0
        epoch_batches = 0
        
        for idx, noise in enumerate(train_loader):
            # 支持两种 DataLoader 返回形式：
            # 1) 旧版：只返回 noise tensor -> 使用仿真生成器生成 clean_spectra（兼容）
            # 2) 新版：返回 (noise_tensor, clean_tensor) -> 直接使用来自 .mat 的 clean（cube）
            if isinstance(noise, (list, tuple)):
                noise_tensor, clean_tensor = noise
                noise = np.squeeze(noise_tensor.numpy())
                clean_spectra = np.squeeze(clean_tensor.numpy())
            else:
                noise = np.squeeze(noise.numpy())
                spectra_num, spec = noise.shape
                clean_spectra = gen_train.generator(spec, spectra_num)
                # 转置（generator 返回 shape -> (spec, spectra_num)）
                clean_spectra = clean_spectra.T
            # noisy_spectra = clean + noise
            noisy_spectra = clean_spectra + noise
            # 定义输入输出
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            # 进行预处理, dct变换
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            # ===== 修改：按样本最大绝对值做归一化（保持样本一致性，便于在验证/预测阶段复现） =====
            eps = 1e-8
            # 每个样本的缩放因子（基于 input_coef 的最大绝对值），形状 (spectra_num,)
            scales = np.maximum(np.max(np.abs(input_coef), axis=1), eps)
            input_coef = input_coef / scales[:, None]
            output_coef = output_coef / scales[:, None]
            # 打印调试信息（输出缩放统计）
            if idx % config.print_freq == 0:
                print(f"[DEBUG] batch {idx} sample_scale mean {scales.mean():.3e}, "
                      f"min {scales.min():.3e}, max {scales.max():.3e}")
            # ===== 归一化结束 =====
            # reshape 成3维度
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            # 转换为tensor
            input_coef = torch.from_numpy(input_coef)
            output_coef = torch.from_numpy(output_coef)
            # 转换为float类型
            input_coef = input_coef.type(torch.FloatTensor)
            output_coef = output_coef.type(torch.FloatTensor)
            if config.use_gpu and torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)
            global_step += 1
            model.train()
            optimizer.zero_grad()
            preds = model(input_coef)
            train_loss = criterion(preds, output_coef)
            train_loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += train_loss.item()
            epoch_batches += 1
            
            # 将训练损失和验证损失写入日志
            writer.add_scalar('train loss', train_loss.item(), global_step=global_step)
            # 每隔xx步打印训练loss
            if idx % config.print_freq == 0:
                print('epoch {}, batch {}/{}, global step {}, train loss = {}'.format(
                    epoch+1, idx, len(train_loader)-1, global_step, train_loss.item()))
            
            # 每50个batch显示一次进度
            if idx % 50 == 0 and idx > 0:
                print('Epoch {} progress: {}/{} batches completed'.format(
                    epoch+1, idx, len(train_loader)-1))
        
        # 打印本轮epoch的平均损失
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        print(f'Epoch {epoch+1}/{config.max_epoch} completed. Average loss: {avg_epoch_loss:.6f}')

        # 保存模型参数,多GPU一定要用module.state_dict()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'global_step': global_step, 'loss': train_loss.item()}
        model_file = os.path.join(save_model_path, str(global_step) + '.pt')
        torch.save(state, model_file)
        """
        加载时使用如下代码 
        state = torch.load(save_model_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        """
        # 验证：需使用eval()方法固定BN
        model.eval()
        valid_loss = 0
        for idx_v, noise in enumerate(valid_loader):
            noise = np.squeeze(noise.numpy())
            spectra_num, spec = noise.shape
            # print(noise.shape)
            clean_spectra = gen_valid.generator(spec, spectra_num)
            # 转置
            clean_spectra = clean_spectra.T
            noisy_spectra = clean_spectra + noise
            # 定义输入输出
            input_coef, output_coef = np.zeros(np.shape(noisy_spectra)), np.zeros(np.shape(noise))
            # 进行预处理, dct变换
            for index in range(spectra_num):
                input_coef[index, :] = dct(noisy_spectra[index, :], norm='ortho')
                output_coef[index, :] = dct(noise[index, :], norm='ortho')
            # ===== 与训练一致：按样本最大值归一化 =====
            eps = 1e-8
            scales_v = np.maximum(np.max(np.abs(input_coef), axis=1), eps)
            input_coef = input_coef / scales_v[:, None]
            output_coef = output_coef / scales_v[:, None]
            # reshape 成3维度
            input_coef = np.reshape(input_coef, (-1, 1, spec))
            output_coef = np.reshape(output_coef, (-1, 1, spec))
            # 转换为tensor
            input_coef = torch.from_numpy(input_coef)
            output_coef = torch.from_numpy(output_coef)
            # float tensor
            input_coef = input_coef.type(torch.FloatTensor)
            output_coef = output_coef.type(torch.FloatTensor)
            if config.use_gpu and torch.cuda.is_available():
                input_coef, output_coef = input_coef.cuda(), output_coef.cuda()
                input_coef.to(device)
                output_coef.to(device)
            preds_v = model(input_coef)
            valid_loss += criterion(preds_v, output_coef).item()
            # 注意：若想在外部对预测值做反归一化保存结果，需要把 preds_v * scales_v[:,None] 恢复
        valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar('valid loss', valid_loss, global_step=global_step)
        # 一个epoch完成后调整学习率
        schedual.step()


def test(config):
    print('testing...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 加载模型参数
    if config.use_gpu and torch.cuda.is_available():
        state = torch.load(model_file)
    else:
        state = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    else:
        model = model.cpu()
        print('using cpu...')
    # 读取测试数据集
    filenames = os.listdir(config.test_data_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 测试数据的绝对路径
            name = config.test_data_root + '/' + file
            # 加载测试数据
            try:
                tmp = sio.loadmat(name)
            except NotImplementedError:
                import h5py
                with h5py.File(name, 'r') as f:
                    # 读取所有数据集
                    tmp = {key: np.array(f[key]) for key in f.keys()}
            inpts, inptr = np.array(tmp[config.test_varible[0]]), np.array(tmp[config.test_varible[1]])
            inpts, inptr = inpts.T, inptr.T
            # s-simulated仿真数据, r-real实际数据
            nums, spec = inpts.shape
            numr, _ = inptr.shape
            
            # 保存原始数据用于后续计算去噪结果
            original_inpts = inpts.copy()
            original_inptr = inptr.copy()
            
            # DCT 变换
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            for idx in range(numr):
                inptr[idx, :] = dct(np.squeeze(inptr[idx, :]), norm='ortho')
            # ===== 与训练保持一致的按样本最大值归一化 =====
            eps = 1e-8
            # 对 inpts 和 inptr 分别按样本最大值归一
            def normalize_batch(arr):
                scales = np.maximum(np.max(np.abs(arr), axis=1), eps)
                return (arr / scales[:, None], scales)
            inpts_norm, scales_s = normalize_batch(inpts)
            inptr_norm, scales_r = normalize_batch(inptr)
            print(f'[DEBUG] File: {file}, Simulated data scales range: [{np.min(scales_s):.6f}, {np.max(scales_s):.6f}], mean: {np.mean(scales_s):.6f}')
            print(f'[DEBUG] File: {file}, Real data scales range: [{np.min(scales_r):.6f}, {np.max(scales_r):.6f}], mean: {np.mean(scales_r):.6f}')
            # 用归一化后的输入做预测
            inpts, inptr = inpts_norm, inptr_norm
            # 转换为3-D tensor
            inpts, inptr = np.array([inpts]).reshape((nums, 1, spec)), np.array([inptr]).reshape((numr, 1, spec))
            inpts, inptr = torch.from_numpy(inpts), torch.from_numpy(inptr)
            # inptt-total
            inptt = torch.cat([inpts, inptr], dim=0)
            # 划分小batch批量测试
            test_size = 32
            group_total = torch.split(inptt, test_size)
            # 存放测试结果
            predt = []
            # preds, predr = [], []
            for i in range(len(group_total)):
                xt = group_total[i]
                if config.use_gpu and torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                predt.append(yt)
            predt = torch.cat(predt, dim=0)
            predt = predt.numpy()
            predt = np.squeeze(predt)
            preds, predr = predt[:nums, :], predt[nums:, :]
            print(f'[DEBUG] File: {file}, Model predictions - Simulated range: [{np.min(preds):.6f}, {np.max(preds):.6f}], Real range: [{np.min(predr):.6f}, {np.max(predr):.6f}]')
            # 反归一化并IDCT，然后用原始信号减去预测噪声得到去噪结果
            for idx in range(nums):
                noise_pred_s = idct(np.squeeze(preds[idx, :] * scales_s[idx]), norm='ortho')
                preds[idx, :] = original_inpts[idx, :] - noise_pred_s
            for idx in range(numr):
                noise_pred_r = idct(np.squeeze(predr[idx, :] * scales_r[idx]), norm='ortho')
                predr[idx, :] = original_inptr[idx, :] - noise_pred_r
            
            print(f'[DEBUG] File: {file}, Final results - Simulated data range: [{np.min(preds):.6f}, {np.max(preds):.6f}], Real data range: [{np.min(predr):.6f}, {np.max(predr):.6f}]')
            # 计算平均相关性（去噪结果与原始含噪声信号的相关性）
            sim_corrs = [np.corrcoef(original_inpts[idx, :], preds[idx, :])[0, 1] for idx in range(nums)]
            real_corrs = [np.corrcoef(original_inptr[idx, :], predr[idx, :])[0, 1] for idx in range(numr)]
            print(f'[DEBUG] File: {file}, Denoised vs Original correlations - Simulated mean: {np.mean(sim_corrs):.6f}, range: [{np.min(sim_corrs):.6f}, {np.max(sim_corrs):.6f}]')
            print(f'[DEBUG] File: {file}, Denoised vs Original correlations - Real mean: {np.mean(real_corrs):.6f}, range: [{np.min(real_corrs):.6f}, {np.max(real_corrs):.6f}]')

            tmp['preds'], tmp['predr'] = preds.T, predr.T
            # 获取存放测试结果目录位置
            test_dir = test_result_dir(config)
            # 新的绝对文件名
            filename = os.path.join(test_dir, "".join(file))
            # 将测试结果保存进测试文件夹
            sio.savemat(filename, tmp)


def predict(config):
    print('predicting...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    save_model_path = save_model_dir(config)
    model_file = os.path.join(save_model_path, str(config.global_step) + '.pt')
    # 加载模型参数
    if config.use_gpu and torch.cuda.is_available():
        state = torch.load(model_file)
    else:
        state = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    else:
        model = model.cpu()
        print('using cpu...')
    # 定义滑动窗口滤波器

    # def moving_average(data, n):
    #     return np.convolve(data, np.ones(n, )/n, mode='same')

    # 读取需要处理的txt文件
    filenames = os.listdir(config.predict_root)
    i = 0
    fig = plt.figure()

    for file in filenames:
        if os.path.splitext(file)[1] == '.txt':
            # 测试数据的绝对路径
            name = config.predict_root + '/' + file
            new_name = config.predict_root + '/dn_' + file
            # 加载测试数据
            tmp = np.loadtxt(name).astype(np.float)
            wave, x = tmp[:, 0], tmp[:, 1]
            # # 数据预处理 滑动窗口滤波+DCT变换
            # xs = moving_average(x, 3)
            # err = x - xs
            # 作为输入数据
            coe_dct = dct(x, norm='ortho')
            # ===== 与训练一致：按样本最大值归一化并在预测后乘回 =====
            eps = 1e-8
            scale = max(np.max(np.abs(coe_dct)), eps)
            coe_dct_norm = coe_dct / scale
            print(f'[DEBUG] File: {file}, DCT scale: {scale:.6f}, DCT coeff range: [{np.min(coe_dct):.6f}, {np.max(coe_dct):.6f}]')
            # 更改shape
            inpt = coe_dct_norm.reshape(1, 1, -1)
             # 转换为torch tensor
            inpt = torch.from_numpy(inpt).float()
             # 预测结果
            if config.use_gpu and torch.cuda.is_available():
                inpt = inpt.cuda()
            yt = model(inpt).detach().cpu().numpy()
            yt = yt.reshape(-1, )
            print(f'[DEBUG] File: {file}, Model output range: [{np.min(yt):.6f}, {np.max(yt):.6f}], mean: {np.mean(yt):.6f}, std: {np.std(yt):.6f}')
            # 反归一化并 idct 变换
            yt = yt * scale
            noise = idct(yt, norm='ortho')
            Y = x - noise
            print(f'[DEBUG] File: {file}, Final result - Original range: [{np.min(x):.6f}, {np.max(x):.6f}], Denoised range: [{np.min(Y):.6f}, {np.max(Y):.6f}]')
            print(f'[DEBUG] File: {file}, Correlation between original and denoised: {np.corrcoef(x, Y)[0,1]:.6f}')
            print(f'[DEBUG] File: {file}, Noise range: [{np.min(noise):.6f}, {np.max(noise):.6f}]')
            denoised = np.array([wave, Y])
            np.savetxt(new_name, denoised, delimiter='\t')
            i = i + 1
            plt.subplot(3, 3, i)
            plt.plot(wave, x)
            plt.plot(wave, Y)
    plt.show()
    print('Prediction completed for all .txt files')

def load_mat_file(filename):
    """Load .mat file, automatically handling v7.3 format"""
    try:
        return sio.loadmat(filename)
    except NotImplementedError:
        with h5py.File(filename, 'r') as f:
            return {key: np.array(f[key]) for key in f.keys()}

def amplitude_correction(original, denoised):
    """更保守的幅度校正策略：
    - 在去噪幅度非常小或原始幅度为0时跳过校正；
    - 计算相关性 corr：若 corr < corr_thresh 则仅做均值对齐，不放大量级；
    - 缩放因子被裁剪到 [min_scale, max_scale]，避免极端放大；
    - 返回与 original 相同 dtype。
    """
    # 参数（可根据需要调整）
    eps = 1e-8
    corr_thresh = 0.4     # 相关性阈值，低于此值不执行幅度放大
    min_scale = 0.8       # 最小允许缩放
    max_scale = 1.25      # 最大允许缩放

    # 计算幅度
    orig_amplitude = np.max(original) - np.min(original)
    denoised_amplitude = np.max(denoised) - np.min(denoised)

    # 如果去噪幅度太小或原始幅度为0，跳过幅度校正
    if denoised_amplitude < eps or orig_amplitude == 0:
        # 保证返回与输入同类型
        print(f"[DEBUG][amplitude_correction] skip: tiny amplitude (den={denoised_amplitude:.3e}, orig={orig_amplitude:.3e})")
        return denoised.astype(original.dtype)

    # 计算相关性（保护性捕获异常）
    try:
        corr = np.corrcoef(original.ravel(), denoised.ravel())[0, 1]
    except Exception:
        corr = 0.0

    # 计算缩放因子
    amplitude_scale = orig_amplitude / denoised_amplitude

    # 裁剪缩放因子到保守范围
    clipped_scale = float(np.clip(amplitude_scale, min_scale, max_scale))

    # 决策：若相关性很低，则不要放大量级（只做均值对齐），同时记录信息
    if corr < corr_thresh:
        # 只做均值对齐，不放大
        corrected = denoised + (np.mean(original) - np.mean(denoised))
        print(f"[DEBUG][amplitude_correction] low corr={corr:.4f} -> skip scaling, mean-align only. orig_amp={orig_amplitude:.3f}, den_amp={denoised_amplitude:.3f}")
        return corrected.astype(original.dtype)

    # 否则应用裁剪后的缩放并对齐均值
    corrected = denoised * clipped_scale
    corrected = corrected + (np.mean(original) - np.mean(corrected))

    print(f"[DEBUG][amplitude_correction] corr={corr:.4f}, scale_requested={amplitude_scale:.3f}, scale_used={clipped_scale:.3f}")
    return corrected.astype(original.dtype)

def batch_predict(config):
    print('batch predicting...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 加载模型参数
    if config.use_gpu and torch.cuda.is_available():
        state = torch.load(model_file)
    else:
        state = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if config.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    else:
        model = model.cpu()
        print('using cpu...')
    # 读取测试数据集
    filenames = os.listdir(config.predict_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 测试数据的绝对路径
            name = config.predict_root + '/' + file
            # 加载测试数据
            try:
                tmp = sio.loadmat(name)
            except NotImplementedError:
                with h5py.File(name, 'r') as f:
                    # 读取所有数据集
                    tmp = {key: np.array(f[key]) for key in f.keys()}
            inpts = np.array(tmp['cube'])
            inpts = inpts.T
            nums, spec = inpts.shape
            print(f'Data shape: {inpts.shape}')
            
            # 保存原始数据用于后续校正
            original_spectra = inpts.copy()
            
            # DCT变换和归一化
            scales = []  # 保存每个样本的尺度因子
            normalized_inputs = []
            
            for idx in range(nums):
                # DCT变换
                coe_dct = dct(np.squeeze(inpts[idx, :]), norm='ortho')
                
                # 归一化
                eps = 1e-8
                scale = max(np.max(np.abs(coe_dct)), eps)
                coe_dct_norm = coe_dct / scale
                
                # 保存尺度因子和归一化数据
                scales.append(scale)
                normalized_inputs.append(coe_dct_norm)
                
                print(f'[DEBUG] File: {file}, Sample {idx}, DCT scale: {scale:.6f}, DCT coeff range: [{np.min(coe_dct):.6f}, {np.max(coe_dct):.6f}]')
            
            # 转换为3D张量
            normalized_inputs = np.array(normalized_inputs).reshape((nums, 1, spec))
            normalized_inputs = torch.from_numpy(normalized_inputs).float()
            
            # 划分小batch批量测试
            test_size = 32
            group_total = torch.split(normalized_inputs, test_size)
            # 存放测试结果
            preds = []
            for i in range(len(group_total)):
                xt = group_total[i]
                if config.use_gpu and torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                preds.append(yt)
            preds = torch.cat(preds, dim=0)
            preds = preds.numpy()
            preds = np.squeeze(preds)
            print(f'[DEBUG] File: {file}, Model predictions shape: {preds.shape}, range: [{np.min(preds):.6f}, {np.max(preds):.6f}], mean: {np.mean(preds):.6f}, std: {np.std(preds):.6f}')
            
            # 反归一化和IDCT变换
            denoised_spectra = np.zeros_like(original_spectra)
            
            for idx in range(nums):
                # 反归一化
                pred_scaled = preds[idx, :] * scales[idx]
                
                # IDCT变换得到预测的噪声
                noise_pred = idct(np.squeeze(pred_scaled), norm='ortho')
                
                # 用原始信号减去预测噪声得到去噪结果
                denoised_temp = original_spectra[idx, :] - noise_pred
                
                # 计算校正前的相关性
                corr_before = np.corrcoef(original_spectra[idx, :], denoised_temp)[0, 1]
                print(f'[DEBUG] File: {file}, Sample {idx}, Before amplitude correction - Correlation: {corr_before:.6f}, Denoised range: [{np.min(denoised_temp):.6f}, {np.max(denoised_temp):.6f}]')
                
                # 应用幅度校正
                denoised_spectra[idx, :] = amplitude_correction(
                    original_spectra[idx, :], 
                    denoised_temp
                )
                
                # 计算校正后的相关性
                corr_after = np.corrcoef(original_spectra[idx, :], denoised_spectra[idx, :])[0, 1]
                print(f'[DEBUG] File: {file}, Sample {idx}, After amplitude correction - Correlation: {corr_after:.6f}, Final range: [{np.min(denoised_spectra[idx, :]):.6f}, {np.max(denoised_spectra[idx, :]):.6f}]')
            
            tmp['preds'] = denoised_spectra.T
            # 获取存放测试结果目录位置
            test_dir = test_result_dir(config)
            # 新的绝对文件名
            filename = os.path.join(test_dir, "".join(file))
            # 将测试结果保存进测试文件夹，过滤掉以__开头的键以避免警告
            save_dict = {k: v for k, v in tmp.items() if not k.startswith('__')}
            sio.savemat(filename, save_dict)
            print(f'Prediction completed and saved to {filename}')

def main(config):
    check_dir(config)
    if config.is_training:
        train(config)
    if config.is_testing:
        test(config)
    if config.is_predicting:
        predict(config)
    if config.is_batch_predicting:
        batch_predict(config)


if __name__ == '__main__':
    import sys
    opt = DefaultConfig()
    
    # 解析命令行参数
    kwargs = {}
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            key, value = arg[2:].split('=', 1)
            # 尝试将值转换为适当的类型
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # 保持为字符串
            kwargs[key] = value
    
    # 更新配置参数
    if kwargs:
        opt._parse(kwargs, opt)
    
    main(opt)
