import torch
import colorama

colorama.init()

def set_up_cpu(use_gpu):

    if use_gpu:
        gpu_available = torch.cuda.device_count() > 0
        if gpu_available:
            gpu_dev = torch.device('cuda:0')
        else:
            gpu_dev = None
    else:
        gpu_available = False
        gpu_dev = None

    return (gpu_available, gpu_dev)

def color_print(msg, type_=None):
    if type_ == 'error':
        print(colorama.Fore.RED + msg)
    elif type_ == 'success':
        print(colorama.Fore.GREEN + msg)
    else:
        print(msg)
    print(colorama.Style.RESET_ALL)

