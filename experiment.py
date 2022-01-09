import os
import functools
import haiku as hk
import jax
import pickle
import pandas as pd
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from lazy import lazy

from data import DataLoader
from monitor import Monitor
from config import configurable
import models
import summary
import mpi4py.MPI as MPI
import mpi4jax
import mpiutils


class Experiment:

    def __init__(self, config):
        print('Initializing experiment')
        self.config = config

        if config.evaluation.frequency % config.summary.frequency != 0:
            raise ValueError('Evaluation frequency must be multiple of summary frequency.')

        # We use MPI instead of pmap due to varying dataset shapes
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        self.data_loader = DataLoader(self.rank,
                                      self.mpi_size,
                                      config.training,
                                      config.evaluation,
                                      config.data)

        self.forward = hk.transform(self._forward_fn)
        self.summary = summary.SummaryLog('experiment')
        self.monitor = Monitor()

        key = jax.random.PRNGKey(self.config.seed)
        self._key, init_key = jax.random.split(key)
        self._train_input = self.data_loader.build_train_dataset()
        init_images, init_labels = next(self._train_input)

        self._params = self.forward.init(init_key, init_images[0], init_labels[0])
        opt_init, self.opt_apply = self._build_optimizer()
        self._opt_state = opt_init(self._params)

        self.restore()
        self._params = mpiutils.tree_bcast(self._params, root=0, comm=self.comm)

        optim_type = self.config.optimizer.type
        if optim_type == 'sgd':
            self.grad_meta_loss_fn = jax.value_and_grad(self._outer_loss)
            self.compute_grads = self._sgd_grads
        elif optim_type == 'es':
            noise_std = self.config.optimizer.noise_std
            self.compute_grads = functools.partial(self._es_grads, noise_std=noise_std)
        else:
            raise ValueError('Invalid optimizer type')

        self._update_func = jax.jit(self._update_func)
        self._evaluate_dataset = jax.jit(self._evaluate_dataset)

    def train(self):
        for i in range(self.config.training.iterations):
            if self.monitor.should_quit:
                break
            self._key, step_key = jax.random.split(self._key)
            scalars = self.step(i, step_key)
            if i % self.config.summary.frequency == 0:
                summary = dict(optim_step=i, **scalars,
                               **self.summary.create(1))
                if i % self.config.evaluation.frequency == 0:
                    summary.update(self.evaluate(log_to_wandb=False))
                wandb.log(summary)
            if (i + 1) % self.config.checkpoint.frequency == 0:
                if self.config.checkpoint.with_step:
                    self.checkpoint(i + 1)
                else:
                    self.checkpoint()
        self.checkpoint()

    @lazy
    def _eval_datasets(self):
        return [self.data_loader.build_eval_dataset(name)
                for name in self.config.evaluation.datasets]

    def _create_eval_dataframe(self, loss, accuracy):
        column_names = ['batch', 'inner_step', 'loss', 'accuracy']
        batch, traj_len = loss.shape
        inner_step = np.arange(traj_len).repeat(batch).reshape(traj_len, batch).T
        data = np.stack([inner_step, loss, accuracy], axis=-1)
        flat_data = summary.flatten_batched_array(data)
        df = pd.DataFrame(flat_data, columns=column_names)
        return df

    def evaluate(self, log_to_wandb=True):
        log = {}
        dfs = []
        rng = jax.random.PRNGKey(self.config.evaluation.seed)
        rng = jax.random.fold_in(rng, self.rank)
        dataset_names = self.config.evaluation.datasets
        for name, dataset in zip(dataset_names, self._eval_datasets):
            images, labels = next(dataset)
            eval_results = self._evaluate_dataset(self._params, rng, images, labels)
            vloss, vaccuracy, loss, accuracy, final_loss, final_accuracy = eval_results

            if self.rank == 0:
                df = self._create_eval_dataframe(vloss, vaccuracy)
                dfs.append(df)
                log[f'eval_loss_{name}'] = np.asscalar(loss)
                log[f'eval_accuracy_{name}'] = np.asscalar(accuracy)
                log[f'eval_final_loss_{name}'] = np.asscalar(final_loss)
                log[f'eval_final_accuracy_{name}'] = np.asscalar(final_accuracy)

        if self.rank == 0:
            path = os.path.join(wandb.run.dir, 'evaluation.ftr')
            df_evaluation = pd.concat(dfs, keys=dataset_names, names=['dataset'])
            df_evaluation.reset_index().to_feather(path)
            wandb.save(path)
            if log_to_wandb:
                log['optim_step'] = 0
                wandb.log(log)

        return log

    def _correct(self, logits, labels):
        correct = (jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)).astype(jnp.float32)
        return correct

    def _accuracy(self, logits, labels):
        return self._correct(logits, labels).mean()

    def _evaluate_dataset(self, params, rng, images, labels):
        # Vmap for `count' dataset evaluations
        fwd = jax.vmap(self.forward.apply, (None, 0, 0, 0))
        rng = jax.random.split(rng, images.shape[0])

        logits = fwd(params, rng, images, labels)
        # Reduce over learning algorithm batch dim
        loss = self._cross_entropy(logits, labels).mean(axis=-1)
        correct = self._correct(logits, labels).mean(axis=-1)

        assert loss.shape == images.shape[:2]
        assert correct.shape == images.shape[:2]

        all_loss, token = mpi4jax.gather(loss, root=0, comm=self.comm)
        all_correct, token = mpi4jax.gather(correct, root=0, comm=self.comm, token=token)

        if self.rank == 0:
            all_loss = all_loss.reshape(-1, loss.shape[-1])
            all_correct = all_correct.reshape(-1, correct.shape[-1])

        loss = all_loss.mean()
        accuracy = all_correct.mean()
        final_loss = all_loss[:, -1].mean()
        final_accuracy = all_correct[:, -1].mean()

        return all_loss, all_correct, loss, accuracy, final_loss, final_accuracy

    def _forward_fn(self, inputs, labels):
        model_config = self.config.model
        input_shape = inputs.shape[2:]
        output_size = labels.shape[-1]

        if model_config.type == 'VSMLRNN':
            vsml_rnn_config = model_config.vsml_rnn
            layer_specs = [models.create_spec(cfg)
                           for cfg in vsml_rnn_config.layer_specs]
            layer_specs = models.complete_specs(layer_specs, input_shape, output_size)
            model = models.VSMLRNN(layer_specs=layer_specs,
                                   loss_func=self._inner_loss)
        elif model_config.type == 'MetaRNN':
            model = models.MetaRNN(loss_func=self._inner_loss,
                                   output_size=output_size)
        elif model_config.type == 'HebbianFW':
            model = models.HebbianFW(loss_func=self._inner_loss,
                                     output_size=output_size,
                                     input_shape=input_shape)
        elif model_config.type == 'SGD':
            model = models.SGD(loss_func=self._inner_loss,
                               output_size=output_size)
        elif model_config.type == 'FWP':
            model = models.FWP(loss_func=self._inner_loss,
                               output_size=output_size)
        elif model_config.type == 'FWMemory':
            model = models.FWMemory(loss_func=self._inner_loss,
                                    output_size=output_size)
        else:
            raise ValueError(f'Invalid model type {model_config.type}')

        return model(inputs, labels)

    def _build_optimizer(self):
        optimizer_config = self.config.optimizer
        optional = []
        if optimizer_config.clip_gradnorm:
            optional.append(optax.clip_by_global_norm(optimizer_config.clip_gradnorm))
        optimizer = optax.chain(
            *optional,
            optax.scale_by_adam(**optimizer_config.kwargs),
            optax.scale(-optimizer_config.lr))
        return optimizer

    @mpiutils.only_rank(0)
    def checkpoint(self, step=None):
        if step is not None:
            name = f'model-{step}.chkp'
        else:
            name = 'model.chkp'
        path = os.path.join(wandb.run.dir, name)
        data = dict(params=self._params, opt_state=self._opt_state)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        wandb.save(path)

    @configurable('checkpoint.restore')
    def restore(self, run_path: str, file_name: str):
        if not run_path:
            return
        if self.rank == 0:
            api = wandb.Api()
            run = api.run(run_path)
            run.file(file_name).download(replace=True).close()
        self.comm.barrier()
        with open(file_name, 'rb') as fb:
            data = pickle.load(fb)
        params = hk.data_structures.merge(self._params, data['params'])
        self._params = params

    def step(self, global_step, rng):
        images, labels = next(self._train_input)

        self._params, self._opt_state, scalars = self._update_func(
            self._params,
            self._opt_state,
            global_step,
            images,
            labels,
            rng)

        scalars = {k: np.asscalar(v) for k, v in scalars.items()}
        return scalars

    def _sgd_grads(self, params, images, labels, rng):
        rng = jax.random.fold_in(rng, self.rank)

        seq_count = self.config.training.population_size // self.mpi_size
        grad_meta_loss_fn = jax.vmap(self.grad_meta_loss_fn, in_axes=(None, 0, 0, 0))
        rng = jax.random.split(rng, seq_count)
        loss, grads = grad_meta_loss_fn(params, images, labels, rng)

        grads = jax.tree_map(functools.partial(jnp.mean, axis=0), grads)
        loss = jnp.mean(loss, axis=0)

        grads = mpiutils.tree_all_reduce(grads, op=MPI.SUM, comm=self.comm)
        loss, _ = mpi4jax.allreduce(loss, op=MPI.SUM, comm=self.comm)
        return grads, loss

    def _update_func(self, params, opt_state, global_step, images, labels, rng):
        grads, loss = self.compute_grads(params, images, labels, rng)

        updates, opt_state = self.opt_apply(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, dict(meta_train_loss=loss)

    def _cross_entropy(self, logits, labels):
        return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)

    def _inner_loss(self, logits, labels, axis=None):
        return self._cross_entropy(logits, labels).mean(axis=axis)

    def _outer_loss(self, params, images, labels, rng):
        logits = self.forward.apply(params, rng, images, labels)
        if self.config.training.loss_type == 'last':
            logits = logits[-1]
            labels = labels[-1]
        loss = self._inner_loss(logits, labels) / self.mpi_size
        return loss

    def _es_eval(self, params, images, labels, rng, noise_std):
        # Extract shapes
        treedef = jax.tree_structure(params)
        shapes = jax.tree_map(lambda p: jnp.array(p.shape), params)

        # Random keys
        rng, param_rng = jax.random.split(rng)
        keys = jax.tree_unflatten(treedef, jax.random.split(param_rng, treedef.num_leaves))

        # Generate noise
        noise = jax.tree_multimap(jax.random.normal, keys, shapes)
        scaled_noise = jax.tree_map(lambda x: x * noise_std, noise)

        # Antithetic sampling
        params_pos = jax.tree_multimap(jnp.add, params, scaled_noise)
        params_neg = jax.tree_multimap(jnp.subtract, params, scaled_noise)

        # TODO Multiple evaluations via vmap
        loss_pos = self._outer_loss(params_pos, images, labels, rng)
        loss_neg = self._outer_loss(params_neg, images, labels, rng)

        es_factor = (loss_pos - loss_neg) / (2 * noise_std ** 2)
        mean_loss = (loss_pos + loss_neg) / 2

        grads = jax.tree_map(lambda x: x * es_factor, scaled_noise)

        return grads, mean_loss

    def _es_grads(self, params, images, labels, rng, noise_std):
        rng = jax.random.fold_in(rng, self.rank)

        seq_count = self.config.training.population_size // self.mpi_size
        v_es_eval = jax.vmap(self._es_eval, in_axes=(None, 0, 0, 0, None))
        rng = jax.random.split(rng, seq_count)
        grads, loss = v_es_eval(params, images, labels, rng, noise_std)

        grads = jax.tree_map(functools.partial(jnp.mean, axis=0), grads)
        loss = jnp.mean(loss, axis=0)

        grads = mpiutils.tree_all_reduce(grads, op=MPI.SUM, comm=self.comm)
        loss, _ = mpi4jax.allreduce(loss, op=MPI.SUM, comm=self.comm)

        return grads, loss
