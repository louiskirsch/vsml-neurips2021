import subprocess
import os
import sys

CUDA_STR = 'CUDA_VISIBLE_DEVICES'
GPU_COUNT_STR = 'GPU_COUNT'
RANK_STR = 'OMPI_COMM_WORLD_LOCAL_RANK'

if CUDA_STR not in os.environ and GPU_COUNT_STR not in os.environ:
    raise ValueError(f'Please set {CUDA_STR} or set {GPU_COUNT_STR} env variable')

if CUDA_STR in os.environ:
    avail_devices = list(map(int, os.getenv(CUDA_STR).split(',')))
else:
    avail_devices = list(range(int(os.getenv(GPU_COUNT_STR))))

devices_count = len(avail_devices)
rank = int(os.getenv(RANK_STR))
os.environ[CUDA_STR] = str(avail_devices[rank % devices_count])
subprocess.call(sys.argv[1:])
