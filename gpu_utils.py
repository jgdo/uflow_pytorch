import subprocess
import os

def auto_gpu(index_gpu = None, default_index = '1'):
    if index_gpu is None:
        result = subprocess.run('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'.split(' '), stdout=subprocess.PIPE)
        lines = result.stdout.splitlines()
        lines = [int(line.decode('ascii')) for line in lines] # get memory usage list, index is gpu number
        available_gpus = sorted(range(len(lines)),key=lines.__getitem__) # sort GPU indices by lowest memory usage

        if available_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[0]) # pick GPU with lowest mem usage
            print('Auto-found free GPU idx {}'.format(available_gpus[0]))
        else:
            print('Warning, no free GPU could be found, using default index ' + default_index)
            os.environ['CUDA_VISIBLE_DEVICES'] = default_index

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(index_gpu)
        print('Setting GPU idx {}'.format(index_gpu))
