import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count
from tqdm import tqdm
import torch


def flops_params_fps(model, input_shape=(1, 3, 256, 256)):
    """count flops:G params:M fps:img/s
        input shape tensor[1, c, h, w]
    """
    total_time = []
    with torch.no_grad():
        model = model.cuda().eval()

        X_input = torch.randn(size=input_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model, X_input)
        params = parameter_count(model)

        for i in tqdm(range(100)):
            torch.cuda.synchronize()
            start = time.time()
            output = model(X_input)
            torch.cuda.synchronize()
            end = time.time()
            total_time.append(end - start)
        mean_time = np.mean(np.array(total_time))
        print(model.__class__.__name__)
        print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))

def flops_params_fps_dual(model, RGB_shape=(1, 3, 256, 256), X_shape=(1, 3, 256, 256)):
    """count flops:G params:M fps:img/s
        input shape tensor[1, c, h, w]
    """
    total_time = []
    with torch.no_grad():
        model = model.cuda().eval()
        RGB_input = torch.randn(size=RGB_shape, dtype=torch.float32).cuda()
        X_input = torch.randn(size=X_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model, (RGB_input, X_input))
        params = parameter_count(model)

        for i in tqdm(range(100)):
            torch.cuda.synchronize()
            start = time.time()
            output = model(RGB_input, X_input)
            torch.cuda.synchronize()
            end = time.time()
            total_time.append(end - start)
        mean_time = np.mean(np.array(total_time))
        print(model.__class__.__name__)
        print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))