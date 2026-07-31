import os
import unittest

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import gymnasium
import numpy as np
import ogbench  # noqa: F401  # Registers OGBench environments.


class MazeReproducibilityTest(unittest.TestCase):
    def test_seeded_reset_and_rollout_are_exact(self):
        for env_name in (
            'pointmaze-giant-v0',
            'antmaze-giant-v0',
            'humanoidmaze-giant-v0',
        ):
            with self.subTest(env_name=env_name):
                env = gymnasium.make(env_name)
                action = np.zeros(
                    env.action_space.shape,
                    dtype=env.action_space.dtype,
                )

                observation, info = env.reset(
                    seed=123,
                    options={'task_id': 2},
                )
                first_observations = [observation.copy()]
                first_goal = info['goal'].copy()
                for _ in range(3):
                    first_observations.append(env.step(action)[0].copy())

                observation, info = env.reset(
                    seed=123,
                    options={'task_id': 2},
                )
                second_observations = [observation.copy()]
                second_goal = info['goal'].copy()
                for _ in range(3):
                    second_observations.append(env.step(action)[0].copy())

                np.testing.assert_array_equal(first_goal, second_goal)
                for first, second in zip(
                    first_observations,
                    second_observations,
                ):
                    np.testing.assert_array_equal(first, second)
                env.close()


if __name__ == '__main__':
    unittest.main()
