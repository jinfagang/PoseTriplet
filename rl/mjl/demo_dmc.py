from dm_control import suite
from dm_control import viewer
import numpy as np

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks
from dm_control.locomotion import soccer

from dm_control.composer.variation import distributions
from dm_control import composer
from torch import rand


def print_all_tasks():
    for domain, task in suite.BENCHMARKING:
        print(domain, task)


print_all_tasks()


def simple_demo():
    env = suite.load(domain_name="humanoid", task_name="walk")
    action_spec = env.action_spec()
    print(action_spec)

    def random_policy(time_step):
        del time_step
        return np.random.uniform(
            low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape
        )

    viewer.launch(env, policy=random_policy)


def cmu_walker():
    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020(
        observable_options={"egocentric_camera": dict(enabled=True)}
    )
    arena = corridor_arenas.WallsCorridor(
        wall_gap=3,
        wall_width=distributions.Uniform(2, 3),
        wall_height=distributions.Uniform(2.5, 3.5),
        corridor_width=4,
        corridor_length=30,
    )
    task = corridor_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0),
        target_velocity=3.0,
        physics_timestep=0.005,
        control_timestep=0.03,
    )
    env = composer.Environment(
        task=task,
        time_limit=10,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    action_spec = env.action_spec()
    print(action_spec)

    def random_policy(time_step):
        del time_step
        return np.random.uniform(
            low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape
        )

    viewer.launch(env, policy=random_policy)


def mul_agent_soccer():
    random_state = np.random.RandomState(42)
    print(random_state)

    env = soccer.load(
        team_size=2,
        time_limit=45,
        random_state=random_state,
        disable_walker_contacts=False,
        walker_type=soccer.WalkerType.ANT,
    )
    env.reset()
    action_spec = env.action_spec()
    print(action_spec)

    def random_policy(time_step):
        del time_step
        if isinstance(action_spec, list):
            return [
                np.random.uniform(
                    low=action_s.minimum, high=action_s.maximum, size=action_s.shape
                )
                for action_s in action_spec
            ]
        else:
            return np.random.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
                size=action_spec.shape,
            )
    viewer.launch(env, policy=random_policy)


if __name__ == "__main__":
    # cmu_walker()
    mul_agent_soccer()
