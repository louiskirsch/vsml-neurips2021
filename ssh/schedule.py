#!/usr/bin/env python3
import argparse
import subprocess
import time
import json
import itertools
from contextlib import suppress


parser = argparse.ArgumentParser()
parser.add_argument('--host', action='store', type=str, required=True)
parser.add_argument('--offset', action='store', type=int, default=None)
parser.add_argument('--oversubscribe', action='store', type=int, default=1)
parser.add_argument('--series', action='store', type=int, default=1)
parser.add_argument('--no-mpi', action='store_true')
parser.add_argument('--nodes', action='store', type=int, default=1)
parser.add_argument('--ntasks-per-node', action='store', type=int, default=1)
parser.add_argument('--time', action='store', type=str)
parser.add_argument('--array', action='store', type=str)
parser.add_argument('--export', action='store', type=str)
parser.add_argument('program', nargs=argparse.REMAINDER)
args = parser.parse_args()

assert len(args.program) > 1 and args.program[0] == '--'
program_args = ' '.join(args.program[1:])
# TODO Update base directory for remote execution
code_path = "CODE_BASE_DIR"
# TODO Update venv path
venv_path = '~/path/to/venv/bin/activate'
gpu_blacklist = dict()
now = time.strftime("%Y-%m-%d_%H-%M-%S")
proj_dir = 'vsml'
name = f"{proj_dir}/{now}"

env_args = args.export.split(',') if args.export else []
with suppress(ValueError): env_args.remove('ALL')
env_args = ' '.join(env_args)


def get_free_gpus():
    gpu_json = subprocess.run(['ssh', args.host, '~/.local/bin/gpustat', '--json'], stdout=subprocess.PIPE, check=True)
    gpus_info = json.loads(gpu_json.stdout)['gpus']
    free = (i for i, info in enumerate(gpus_info) if len(info['processes']) == 0)
    blacklist = gpu_blacklist.get(args.host, ())
    return (i for i in free if i not in blacklist)


if args.array:
    a_start, a_end = map(int, args.array.split('-'))
    array = range(a_start, a_end + 1)
else:
    array = range(1)

if args.series > 1:
    array = [array[i:i + args.series] for i in range(0, len(array), args.series)]

if args.offset is None:
    avail_gpus = list(get_free_gpus())
else:
    avail_gpus = list(range(args.offset, 8))

gpus_needed = args.nodes * len(array)
avail_gpu_count = len(avail_gpus) * args.oversubscribe
assert avail_gpu_count >= gpus_needed, \
       f'Not enough GPUs, {gpus_needed} needed, {avail_gpu_count} available.'
gpu_queue = itertools.chain.from_iterable(itertools.repeat(avail_gpus, args.oversubscribe))

print('Syncing code ...')
subprocess.call(f"rsync -az --include 'wandb/settings' --exclude-from .gitignore --exclude '.git/' "
                f". {args.host}:{code_path}/{name}", shell=True)
print('Syncing git ...')
subprocess.call(f"rsync -az --delete .git {args.host}:{code_path}/{proj_dir}", shell=True)

split_commands = [rf"split-window -t runs '{{{i}}} || read' \; select-layout -t runs tiled \;"
                  for i in range(len(array))]
remote_command = r"2>/dev/null tmux new-session -s runs -d '{0} || read' \; " + " ".join(split_commands[1:]) + \
                 r" || tmux " + " ".join(split_commands)
commands = []

for a_job in array:
    devices = ','.join(map(str, itertools.islice(gpu_queue, args.nodes)))
    process_count = args.ntasks_per_node * args.nodes
    if not args.no_mpi:
        mpi_cmd = f'mpirun -n {process_count} --oversubscribe --bind-to none python3 assign_gpu.py'
    else:
        mpi_cmd = f'MPI_COUNT={process_count}'
    create_cmd = lambda job_id: (f'SLURM_ARRAY_TASK_ID={job_id} CUDA_VISIBLE_DEVICES={devices} CUDA_AWARE=1'
                                 f'{env_args} {mpi_cmd} {program_args}')
    try:
        counter = r'\\\$i'
        inner_cmd = f'for i in {{{a_job[0]}..{a_job[-1]}}}; do {create_cmd(counter)}; done'
    except TypeError:
        inner_cmd = create_cmd(a_job)
    commands.append(f'bash -c \\". /etc/profile; . {venv_path}; {inner_cmd} \\"')

print('Launching ...')
subprocess.call(f'ssh {args.host} "cd {code_path}/{name}; '
                f'ln -s ../.git .git; '
                f'sed -i \'/^disabled = true/d\' wandb/settings; '
                f'sed -i \'/^mode = offline/d\' wandb/settings; '
                f'{remote_command.format(*commands)}"', shell=True)
