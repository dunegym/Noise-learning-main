def batch_predict(config):
    print('batch predicting...')
    # build the model
    model = eval("{}(1,1)".format(config.model_name))
    # 获取保存模型路径
    # save_model_path = save_model_dir(config)
    model_file = os.path.join(config.test_model_dir, str(config.global_step) + '.pt')
    # 加载模型参数
    state = torch.load(model_file)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()})
    # model.load_state_dict(state['model'])
    print('Successfully loaded The model saved at global step = {}'.format(state['global_step']))
    # 固定模型
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print('using gpu...')
    # 读取测试数据集
    filenames = os.listdir(config.predict_root)
    for file in filenames:
        if os.path.splitext(file)[1] == '.mat':
            # 测试数据的绝对路径
            name = config.predict_root + '/' + file
            # 加载测试数据
            tmp = sio.loadmat(name)
            inpts = np.array(tmp['cube'])
            inpts = inpts.T
            nums, spec = inpts.shape
            # DCT 变换
            for idx in range(nums):
                inpts[idx, :] = dct(np.squeeze(inpts[idx, :]), norm='ortho')
            # 转换为3-D tensor
            inpts = np.array([inpts]).reshape((nums, 1, spec))
            inpts = torch.from_numpy(inpts)

            # 划分小batch批量测试
            test_size = 32
            group_total = torch.split(inpts, test_size)
            # 存放测试结果
            preds = []
            for i in range(len(group_total)):
                xt = group_total[i]
                if torch.cuda.is_available():
                    xt = xt.cuda()
                yt = model(xt).detach().cpu()
                preds.append(yt)
            preds = torch.cat(preds, dim=0)
            preds = preds.numpy()
            preds = np.squeeze(preds)
            for idx in range(nums):
                preds[idx, :] = idct(np.squeeze(preds[idx, :]), norm='ortho')
            tmp['preds'] = preds.T
            # 获取存放测试结果目录位置
            test_dir = test_result_dir(config)
            # 新的绝对文件名
            filename = os.path.join(test_dir, "".join(file))
            # 将测试结果保存进测试文件夹，过滤掉以__开头的键以避免警告
            save_dict = {k: v for k, v in tmp.items() if not k.startswith('__')}
            sio.savemat(filename, save_dict)