import json
import importlib.metadata
import os
import platform
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, MultiHGCDataset
from utils.env_utils import make_env_and_datasets
from utils.navigation_dataset import NavigationDataset, position_statistics
from utils.evaluation import derive_seed, evaluate
from utils.flax_utils import (
    build_checkpoint_metadata,
    restore_agent,
    save_agent,
)
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('wandb_project', 'CoGHP', 'W&B project name and experiment subdirectory.')
flags.DEFINE_string('wandb_entity', None, 'W&B entity. None uses the account default.',)
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'], 'W&B logging mode.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('run_dir', None, 'Exact artifact directory; must not already exist. Overrides save_dir layout.')
flags.DEFINE_bool('require_gpu', False, 'Fail instead of silently training on CPU.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_bool('eval_only', False, 'Evaluate a restored checkpoint without applying a training update.')
flags.DEFINE_bool(
    'checkpoint_signature_required',
    False,
    'Require checkpoint architecture metadata and reject mismatched restores.',
)
flags.DEFINE_string(
    'experiment_git_commit',
    None,
    'Source git commit recorded in checkpoint metadata.',
)
flags.DEFINE_string(
    'experiment_code_signature',
    None,
    'Hash of the synchronized source files recorded in checkpoint metadata.',
)

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_list('save_steps', [], 'Additional exact checkpoint steps, e.g. 800000,900000,1000000.')
flags.DEFINE_integer('eval_seed', None, 'Base seed for deterministic evaluation. None derives it from --seed.')
flags.DEFINE_integer('train_dataset_seed', None, 'Training dataset RNG seed. None derives it from --seed.')
flags.DEFINE_integer('validation_dataset_seed', None, 'Validation dataset RNG seed. None derives it from --seed.')
flags.DEFINE_integer('environment_seed', None, 'Environment initialization seed. None derives it from --seed.')
flags.DEFINE_bool(
    'fixed_validation_batch',
    True,
    'Reuse the same deterministically sampled validation batch at every logging step.',
)
flags.DEFINE_bool(
    'require_deterministic_gpu',
    True,
    'Require XLA deterministic GPU operations when running on a GPU backend.',
)

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/coghp.py', lock_config=False)


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if FLAGS.require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('--require_gpu was set but JAX did not initialize a GPU backend.')
    save_steps = {int(step) for step in FLAGS.save_steps}
    if any(step <= 0 or step > FLAGS.train_steps for step in save_steps):
        raise ValueError('--save_steps must lie within the training budget.')
    if (
        FLAGS.require_deterministic_gpu
        and jax.default_backend() == 'gpu'
        and '--xla_gpu_deterministic_ops' not in os.environ.get('XLA_FLAGS', '')
    ):
        raise RuntimeError(
            'GPU reproducibility requires XLA_FLAGS to contain '
            '--xla_gpu_deterministic_ops before Python starts. Use the '
            'repository training scripts or set DETERMINISTIC=1.'
        )

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    if FLAGS.run_dir is not None:
        FLAGS.save_dir = os.path.abspath(FLAGS.run_dir)
        os.makedirs(FLAGS.save_dir, exist_ok=False)
    setup_wandb(
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
        group=FLAGS.run_group,
        name=exp_name,
        mode=FLAGS.wandb_mode,
    )

    if FLAGS.run_dir is None:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
        os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Derive independent deterministic random streams. Explicit flags allow an
    # experiment to hold one stream fixed while changing another.
    train_dataset_seed = (
        FLAGS.train_dataset_seed
        if FLAGS.train_dataset_seed is not None
        else derive_seed(FLAGS.seed, 1)
    )
    validation_dataset_seed = (
        FLAGS.validation_dataset_seed
        if FLAGS.validation_dataset_seed is not None
        else derive_seed(FLAGS.seed, 2)
    )
    environment_seed = (
        FLAGS.environment_seed
        if FLAGS.environment_seed is not None
        else derive_seed(FLAGS.seed, 3)
    )
    eval_seed = (
        FLAGS.eval_seed
        if FLAGS.eval_seed is not None
        else derive_seed(FLAGS.seed, 4)
    )

    # Set up environment and dataset.
    config = FLAGS.agent
    if FLAGS.checkpoint_signature_required and not FLAGS.experiment_git_commit:
        raise ValueError(
            '--experiment_git_commit is required when checkpoint signature '
            'validation is enabled.'
        )
    checkpoint_metadata = build_checkpoint_metadata(
        config,
        environment=FLAGS.env_name,
        seed=FLAGS.seed,
        git_commit=FLAGS.experiment_git_commit,
        code_signature=FLAGS.experiment_code_signature,
        resume=FLAGS.restore_path is not None,
        restore_path=FLAGS.restore_path,
    )
    with open(
        os.path.join(FLAGS.save_dir, 'checkpoint_metadata.json'),
        'w',
    ) as f:
        json.dump(checkpoint_metadata, f, indent=2, sort_keys=True)
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config['frame_stack'],
        seed=environment_seed,
    )
    navigation_study = config['agent_name'] == 'navigation_coghp'
    if navigation_study:
        stats = position_statistics(train_dataset)
        config.position_mean = tuple(stats['mean'])
        config.position_scale = tuple(stats['scale'])
        if not hasattr(env.unwrapped, '_goal_tol'):
            raise ValueError('Navigation study requires an explicit environment goal tolerance.')
        config.global_goal_tolerance = float(env.unwrapped._goal_tol)
        with open(os.path.join(FLAGS.save_dir, 'position_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        # Resolve data-dependent semantics before recording flags/signatures.
        checkpoint_metadata = build_checkpoint_metadata(
            config, environment=FLAGS.env_name, seed=FLAGS.seed,
            git_commit=FLAGS.experiment_git_commit, code_signature=FLAGS.experiment_code_signature,
            resume=FLAGS.restore_path is not None, restore_path=FLAGS.restore_path)
        with open(os.path.join(FLAGS.save_dir, 'checkpoint_metadata.json'), 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(get_flag_dict(), f)
        wandb.config.update({'agent': config.to_dict()}, allow_val_change=True)

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'MultiHGCDataset': MultiHGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(
        Dataset.create(seed=train_dataset_seed, **train_dataset),
        config,
        rng=np.random.default_rng(train_dataset_seed),
    )
    if val_dataset is not None:
        val_dataset = dataset_class(
            Dataset.create(seed=validation_dataset_seed, **val_dataset),
            config,
            rng=np.random.default_rng(validation_dataset_seed),
        )
    if navigation_study:
        train_dataset = NavigationDataset(train_dataset, config)
        if val_dataset is not None:
            val_dataset = NavigationDataset(val_dataset, config)

    reproducibility_info = {
        'seed': FLAGS.seed,
        'train_dataset_seed': train_dataset_seed,
        'validation_dataset_seed': validation_dataset_seed,
        'environment_seed': environment_seed,
        'eval_seed': eval_seed,
        'fixed_validation_batch': FLAGS.fixed_validation_batch,
        'require_deterministic_gpu': FLAGS.require_deterministic_gpu,
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'jax_version': jax.__version__,
        'platform': platform.platform(),
        'jax_devices': [str(device) for device in jax.devices()],
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'xla_flags': os.environ.get('XLA_FLAGS'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'dataset_dir': os.environ.get('OGBENCH_DATASET_DIR'),
        'jax_backend': jax.default_backend(),
        'package_versions': {
            name: importlib.metadata.version(name)
            for name in ('jaxlib', 'flax', 'optax', 'mujoco', 'gymnasium')
        },
    }
    with open(os.path.join(FLAGS.save_dir, 'reproducibility.json'), 'w') as f:
        json.dump(reproducibility_info, f, indent=2)

    # Initialize agent.
    # Agent shape initialization must not consume the training sampler's RNG.
    example_batch = train_dataset.dataset.get_subset(np.array([0], dtype=np.int64))
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    if navigation_study:
        counts = {
            name: sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params))
            for name, params in agent.network.params.items()
        }
        with open(os.path.join(FLAGS.save_dir, 'model_parameter_counts.json'), 'w') as f:
            json.dump(dict(by_group=counts, total_stored_parameters=sum(counts.values()),
                           note='Includes target copies and any inactive legacy modules; not an active-compute count.'),
                      f, indent=2)

    # Restore agent and, for this new study, the exact sampler continuation.
    first_update = 1
    if FLAGS.restore_path is not None:
        agent = restore_agent(
            agent,
            FLAGS.restore_path,
            FLAGS.restore_epoch,
            expected_checkpoint_metadata=checkpoint_metadata,
            require_checkpoint_metadata=FLAGS.checkpoint_signature_required,
        )
        if navigation_study and not FLAGS.eval_only:
            state_path = os.path.join(FLAGS.restore_path, f'training_state_{FLAGS.restore_epoch}.json')
            with open(state_path) as f:
                sampler_state = json.load(f)
            if int(agent.network.step) != FLAGS.restore_epoch + 1:
                raise ValueError('Checkpoint update count does not match restore epoch.')
            train_dataset.rng.bit_generator.state = sampler_state['train_rng']
            if val_dataset is not None:
                val_dataset.rng.bit_generator.state = sampler_state['validation_rng']
            first_update = FLAGS.restore_epoch + 1
    if FLAGS.eval_only:
        if FLAGS.restore_path is None or FLAGS.restore_epoch is None:
            raise ValueError('--eval_only requires both --restore_path and --restore_epoch.')
        FLAGS.train_steps = 1
        FLAGS.eval_interval = 1

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    previous_position_probe = None
    for i in tqdm.tqdm(range(first_update, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        if not FLAGS.eval_only:
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)

        # Log metrics.
        if not FLAGS.eval_only and i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                if FLAGS.fixed_validation_batch:
                    validation_rng = np.random.default_rng(validation_dataset_seed)
                else:
                    validation_rng = val_dataset.rng
                val_batch = val_dataset.sample(
                    config['batch_size'],
                    evaluation=True,
                    rng=validation_rng,
                )
                validation_key = jax.random.PRNGKey(
                    derive_seed(validation_dataset_seed, 0)
                )
                validation_loss_key, validation_rollout_key = jax.random.split(
                    validation_key
                )
                _, val_info = agent.total_loss(
                    val_batch,
                    grad_params=None,
                    rng=validation_loss_key,
                )
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                if (
                    hasattr(agent, 'validation_rollout_info')
                    and (
                        config.get('enable_hcoghp_diagnostics', False)
                        or config.get('enable_free_running_validation', False)
                        or config.get('enable_fm_diagnostics', False)
                        or config.get('enable_action_hrm_validation', False)
                    )
                ):
                    rollout_info = agent.validation_rollout_info(
                        val_batch,
                        rng=validation_rollout_key,
                    )
                    train_metrics.update(rollout_info)
                if navigation_study and config['command_content'] == 'position':
                    train_metrics.update(agent.gradient_diagnostics(val_batch))
                    probe = np.asarray(agent.position_probe(val_batch))
                    train_metrics['position_drift/mse_since_previous_log'] = (
                        0.0 if previous_position_probe is None else float(np.mean((probe - previous_position_probe) ** 2)))
                    train_metrics['position_drift/has_previous'] = float(previous_position_probe is not None)
                    previous_position_probe = probe
                    if i % FLAGS.eval_interval == 0 or i in save_steps:
                        np.savez_compressed(os.path.join(FLAGS.save_dir, f'position_probe_{i}.npz'),
                                            latent=probe, observations=val_batch['observations'],
                                            positions=val_batch['high_actor_targets'][..., :2])
            if navigation_study:
                bad = [key for key, value in train_metrics.items() if not np.isfinite(np.asarray(value)).all()]
                if bad:
                    raise FloatingPointError(f'Nonfinite navigation metrics at update {i}: {bad}')
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i % FLAGS.eval_interval == 0:
        # if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    # Keep every checkpoint on the same task/episode set.
                    seed=derive_seed(eval_seed, task_id),
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if not FLAGS.eval_only and (i % FLAGS.save_interval == 0 or i in save_steps):
            save_agent(
                agent,
                FLAGS.save_dir,
                i,
                checkpoint_metadata=checkpoint_metadata,
            )
            if navigation_study:
                sampler_state = dict(train_rng=train_dataset.rng.bit_generator.state,
                                     validation_rng=val_dataset.rng.bit_generator.state if val_dataset is not None else None)
                with open(os.path.join(FLAGS.save_dir, f'training_state_{i}.json'), 'w') as f:
                    json.dump(sampler_state, f)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
