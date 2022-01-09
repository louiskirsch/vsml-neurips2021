#!/usr/bin/env python3
import sys
import subprocess
import time
import os

# TODO Update host and code path
host_name = "SLURM_HOST"
code_path = "my/path"

separator = sys.argv.index('--')
slurm_args = ' '.join(sys.argv[1:separator])
program_args = ' '.join(sys.argv[separator + 1:])

proj_dir = 'vsml'
dependency = ''

resume = int(os.getenv('RESUME', '0'))
if 'REUSE' not in os.environ:
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    directory = f'{code_path}/{proj_dir}/{now}'
    name = now

    print('Syncing code ...')
    sync_code = subprocess.Popen(f"rsync -az --include 'wandb/settings' --exclude-from .gitignore --exclude '.git/' "
                                 f". {host_name}:{directory}", shell=True)

    print('Syncing git ...')
    sync_git = subprocess.Popen(f"rsync -az --delete .git {host_name}:{code_path}/{proj_dir}", shell=True)

    if sync_code.wait() == 0:
        print('✔️ Synced code')
    else:
        print('❌ Failed to sync code')
        sys.exit(1)
    if sync_git.wait() == 0:
        print('✔️ Synced git')
    else:
        print('❌ Failed to sync git')
        sys.exit(2)

    if resume:
        dependency = f'--export=ALL,WANDB_DEPENDENCY={resume}'
else:
    reuse = os.getenv('REUSE')
    name = f'j{reuse}'
    directory = f"$(dirname $(printf {code_path}/{proj_dir}/*/slurm-{reuse}*))"
    if resume:
        dependency = f'--export=ALL,WANDB_DEPENDENCY={reuse}'

sbatch = []
chain_count = int(os.getenv('CHAIN', '1'))
if chain_count > 1:
    for i in range(chain_count):
        if i > 0:
            dependency = '--dependency=afterok:"$id" --export=ALL,WANDB_DEPENDENCY="$id"'
        sbatch.append(f'id=`sbatch --parsable {dependency} {slurm_args} slurm/job.sh {program_args}`')
        sbatch.append('echo "Submitted batch job $id"')
else:
    sbatch.append(f'sbatch {dependency} {slurm_args} slurm/job.sh {program_args}')
sbatch = ' && '.join(sbatch)

print(f"Launching {name} with args {slurm_args} and script args {program_args} ...")
subprocess.call(f"ssh {host_name} 'cd \"{directory}\" && "
                f"ln -s ../.git .git ; "
                f"sed -i \"/^disabled = true/d\" wandb/settings && "
                f"{sbatch}'", shell=True)
