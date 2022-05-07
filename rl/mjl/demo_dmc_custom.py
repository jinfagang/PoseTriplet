import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control import viewer
from dm_control import suite
from dm_control.utils import io as resources
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = 0.04

SUITE = containers.TaggedTasks()
THIS_DIR = os.path.dirname(__file__)
# MODEL_XML_PATH = '/home/jintian/mujoco/model/humanoid/humanoid.xml'
MODEL_XML_PATH = os.path.join(THIS_DIR, "./models/humanoid_h36m/humanoid_h36m_v5.xml")
# MODEL_XML_PATH = "models/mug/mug.xml"


def get_model_and_assets():
    _FILENAMES = [
        "./common/materials.xml",
        "./common/skybox.xml",
        "./common/visual.xml",
        "./common/sky1.png",
        "./common/grass.png",
    ]
    ASSETS = {
        filename: resources.GetResource(
            os.path.join(os.path.dirname(MODEL_XML_PATH), filename)
        )
        for filename in _FILENAMES
    }
    print(ASSETS.keys())
    # return common.read_model(MODEL_XML_PATH), common.ASSETS
    # return resources.GetResource(MODEL_XML_PATH), common.ASSETS
    return resources.GetResource(MODEL_XML_PATH), ASSETS


@SUITE.add("playing")
def my_env(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Robot()
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        # control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics class"""


class Robot(base.Task):
    def __init__(self, random=None):
        """Initializes an instance of `Robot`."""
        super(Robot, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        pass

    def get_observation(self, physics):
        """Returns either the pure state or a set of egocentric features."""
        obs = collections.OrderedDict()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return 0
    
    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return mujoco.action_spec(physics)


def simple_demo():
    env = my_env()
    action_spec = env.action_spec()
    print(action_spec)

    def random_policy(time_step):
        del time_step
        rdm_policy = np.random.uniform(
            # low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape
            low=0, high=1, size=action_spec.shape
        )
        print(rdm_policy)
        return rdm_policy

    viewer.launch(env, policy=random_policy)


if __name__ == "__main__":
    simple_demo()
