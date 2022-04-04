import sys
import torch
from includes import ArgParser, Model, Utils

args = ArgParser.parse_arguments()
gpu_available, gpu_dev = Utils.set_up_cpu(args.use_gpu)
if not gpu_available or gpu_dev == None:
    Utils.color_print('Only GPU processing is supported at the moment!', type_='error')
    sys.exit(-1)
args.gpu = {'gpu_available': gpu_available, 'gpu_dev': gpu_dev}

m1 = Model.M1(
    input_size=[1,5,512,384],
    L_depth=[64,64,128,128,256,256,256,256,512,512,512,512,512,512,512,512],
    kernel_size=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    padding=[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,],
    stride=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,])
if args.gpu['gpu_available']:
    m1.cuda(args.gpu['gpu_dev'])

m1.eval()
x = torch.randn((1,5,512,384), dtype=torch.float32)
if args.gpu['gpu_available']:
    x = x.cuda(args.gpu['gpu_dev'])
x_out = m1(x)
print('input shape', x.size())
print(x_out.size())

