import torch
import subprocess

flag = torch.cuda.is_available()
print(flag)  # 返回true为安装成功

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

# Check CUDA version
cuda_version = torch.version.cuda
print("CUDA Version：", cuda_version)
# Check CuDNN version
cuda_version = torch.backends.cudnn.version()
print("CuDNN Version:", cuda_version)

# 生成environment.txt
cmd = "pip freeze > environment.txt"
environment = subprocess.run(cmd, shell=True)
print("已生成 environment.txt")
