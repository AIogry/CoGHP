"""Position/OTA fields without altering the legacy batch or its RNG stream.

Compact OGBench retains the final observation with valids=0, but marks both
that observation and the preceding action as terminal. Global fragments use
the actual final observation (valids=0), not that preceding action's index.
Incomplete, unsuccessful fragments are masked; early successes remain valid.
"""

import numpy as np


def position_statistics(dataset):
    xy = np.asarray(dataset['observations'][:, :2], dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError('Nonfinite training positions.')
    return dict(mean=xy.mean(axis=0).tolist(),
                scale=np.maximum(xy.std(axis=0), 1e-6).tolist(),
                count=len(xy), rule='population std of all training observations; floor=1e-6')


class GlobalFragmentSampler:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.n = int(config['global_abstraction_steps'])
        self.tolerance = float(config['global_goal_tolerance'])
        if self.n < 1 or self.tolerance <= 0:
            raise ValueError('Positive abstraction length and goal tolerance required.')
        if 'valids' not in dataset or 'next_observations' in dataset:
            raise ValueError('Global sampling requires the verified compact dataset representation.')
        self.ends = np.flatnonzero(np.asarray(dataset['valids']) == 0)
        if not len(self.ends) or self.ends[-1] != len(dataset['observations']) - 1:
            raise ValueError('Compact dataset must retain a final observation for every trajectory.')

    def add_fields(self, batch):
        # Reuse the existing value-goal draw, including its 0.2/0.5/0.3 mixture.
        # No extra random draw and no change to the legacy sampler sequence.
        idxs = np.asarray(batch['indices'], dtype=np.int64)
        goals = np.asarray(batch['value_goals'][..., :2])
        ends = self.ends[np.searchsorted(self.ends, idxs)]
        offsets = np.arange(self.n + 1)
        indices = np.minimum(idxs[:, None] + offsets, ends[:, None])
        positions = self.dataset['observations'][indices, :2]
        within = idxs[:, None] + offsets <= ends[:, None]
        reached = (np.linalg.norm(positions - goals[:, None], axis=-1) <= self.tolerance) & within
        success = reached.any(axis=1)
        first = np.argmax(reached, axis=1)
        endpoint = np.where(success, idxs + first, np.minimum(idxs + self.n, ends))
        valid = success | (idxs + self.n <= ends)
        result = dict(batch)
        result.update(
            global_goals=goals,
            global_observations=self.dataset['observations'][endpoint],
            global_rewards=np.where(success, 0.0, -1.0).astype(np.float32),
            global_masks=(~success).astype(np.float32),
            global_valid=valid.astype(np.float32),
            global_offsets=(endpoint - idxs).astype(np.int32),
        )
        return result


class NavigationDataset:
    """Wrap an unchanged MultiHGCDataset. No independent RNG is consumed."""

    def __init__(self, legacy, config):
        self.legacy = legacy
        self.global_sampler = (GlobalFragmentSampler(legacy.dataset, config)
                               if config['high_weight_source'] == 'global_ota' else None)

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def sample(self, *args, **kwargs):
        batch = self.legacy.sample(*args, **kwargs)
        return self.global_sampler.add_fields(batch) if self.global_sampler else batch
