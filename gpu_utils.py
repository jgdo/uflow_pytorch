import subprocess
import os

def auto_gpu(index_gpu = None):
    if index_gpu is None:
        result = subprocess.run(['nvidia-smi', 'pmon', '-c', '1'], stdout=subprocess.PIPE)
        lines = result.stdout.splitlines()
        lines = [line.decode('ascii') for line in lines]
        elements = [line.split() for line in lines if not line.startswith('#')]
        idx_user = [(int(n[0]), n[7]) for n in elements]  # (idx, mem_util)
        available_gpus = [idx for idx, user in idx_user if user == '-']

        if available_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[0])
            print('Auto-found free GPU idx {}'.format(available_gpus[0]))
        else:
            print('Warning, no free GPU could be found, using default variable')

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(index_gpu)
        print('Setting GPU idx {}'.format(index_gpu))
