import wandb
import mpi4py.MPI as MPI
import yaml
import logging
import os
import config
import numpy as np
import tensorflow as tf

from experiment import Experiment
from config import expand_dot_items, WandbMockConfig, WandbMockSummary, flatten_dot_items, DotDict, GLOBAL_CONFIG


def _warn_new_keys(config, existing_config):
    for k in config.keys() - existing_config.keys():
        if '.mpi_' not in k:
            logging.warning(f'Specified config key {k} does not exist.')


def _merge_config(config, config_files, task_config=None):
    derived_config = {}
    config_files = ['configs/default.yaml'] + config_files
    for cfg in config_files:
        with open(cfg, mode='r') as f:
            new_config = flatten_dot_items(yaml.safe_load(f))
            if len(derived_config) > 0:
                _warn_new_keys(new_config, derived_config)
            derived_config.update(new_config)
    if task_config is not None:
        _warn_new_keys(task_config, derived_config)
        derived_config.update(task_config)
    if config is not None:
        _warn_new_keys(config, derived_config)
        derived_config.update(config)
    derived_config = flatten_dot_items(derived_config)
    return derived_config


def _sync_config(comm, mpi_rank):
    config = wandb.config._items if mpi_rank == 0 else None
    config = comm.bcast(config, root=0)
    if mpi_rank > 0:
        # Create mock wandb objects because we didnt initialize wandb in this process
        wandb.config = WandbMockConfig(config)
        wandb.summary = WandbMockSummary()
        wandb.log = lambda *args, **kwargs: None
    return config


def _update_ranked_config(config: dict, mpi_rank: int):
    # TODO potentially add support for dictionaries within the mpi_split
    for k, v in filter(lambda it: 'mpi_split' in it[0], list(config.items())):
        base_key = k.replace('.mpi_split', '')
        repeat_key = f'{base_key}.mpi_repeat'
        repeat = config.pop(repeat_key, 1)
        idx = mpi_rank // repeat
        selected_option = v[idx % len(v)]
        config[base_key] = selected_option
        del config[k]
    return config


def _save_log_to_wandb():
    if 'SLURM_JOB_ID' in os.environ:
        if 'SLURM_ARRAY_JOB_ID' in os.environ:
            job_id = os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
            wandb.summary.slurm_array_jobid = os.environ['SLURM_ARRAY_JOB_ID']
        else:
            job_id = os.environ['SLURM_JOB_ID']
        wandb.summary.slurm_jobid = job_id
        log_src = f'slurm-{job_id}.out'
        log_dst = os.path.join(wandb.run.dir, f'slurm-{job_id}.txt')
        os.link('./' + log_src, log_dst)
        wandb.save(log_dst)
        with open(f'wandb-run-{job_id}', 'a') as f:
            f.write(f'{wandb.run.id}\n')


def _create_array_task(spec, mpi_rank, array_subset):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    tasks = spec['array']
    if array_subset is not None and len(array_subset) > 0:
        task = tasks[array_subset[task_id - 1]]
    else:
        task = tasks[task_id - 1]
    if mpi_rank == 0:
        logging.info(f'Loading task {task_id}:\n{yaml.dump(task)}')
    tags = task.get('tags', [])
    config_files = task.get('config_files', [])
    config = task.get('config', {})
    config['task_id'] = task_id
    return tags, config_files, config


def _create_grid_task(spec, mpi_rank):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    grid = spec['grid']
    task_count = np.prod([len(ax) for ax in grid])
    if task_id > task_count:
        raise ValueError(f'There are only {task_count} tasks, {task_id} was requested')
    selection = []
    i = task_id - 1  # One based task id
    for ax in grid:
        selection.append(ax[i % len(ax)])
        i //= len(ax)

    if mpi_rank == 0:
        logging.info(f'Loading grid selection {task_id} of {task_count}:\n{yaml.dump(selection)}')

    tags = []
    config_files = []
    config = dict(task_id=task_id)
    for ax in selection:
        tags.extend(ax.get('tags', []))
        config_files.extend(ax.get('config_files', []))
        config.update(flatten_dot_items(ax.get('config', {})))
    return tags, config_files, config


def _setup_config(mpi_rank, config, config_files, array_file, array_subset):
    tags = []
    config_files = config_files or []
    if array_file is not None:
        with open(array_file, mode='r') as f:
            spec = yaml.safe_load(f)
            if 'array' in spec:
                t_tags, t_config_files, t_config = _create_array_task(spec, mpi_rank, array_subset)
            elif 'grid' in spec:
                t_tags, t_config_files, t_config = _create_grid_task(spec, mpi_rank)
            tags.extend(t_tags)
            config_files.extend(t_config_files)
            task_config = t_config
    else:
        task_config = None
    config = _merge_config(config, config_files, task_config)
    return config, tags


def run(args):
    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    tf_log_level = os.environ.get('TF_LOGLEVEL', 'WARN').upper()
    logging.basicConfig(level=log_level)
    tf.get_logger().setLevel(tf_log_level)
    logging.info('Launching')

    # Disable tensorflow GPU support
    tf.config.experimental.set_visible_devices([], "GPU")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        config, tags = _setup_config(rank, args.config, args.config_files, args.array, args.subset)
        tags = tags + args.tags if args.tags else tags
        wandb.init(config=config, tags=tags, job_type=args.job_type)
        _save_log_to_wandb()
        wandb.summary.mpi_size = comm.Get_size()
    config = _sync_config(comm, rank)
    config = _update_ranked_config(config, rank)
    config = expand_dot_items(DotDict(config))
    GLOBAL_CONFIG.update(config)
    experiment = Experiment(config)
    entry_fn = getattr(experiment, config.call)
    entry_fn()


if __name__ == '__main__':
    run(config.parse_args())
