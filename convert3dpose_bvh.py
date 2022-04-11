from video2bvh.bvh_skeleton import humanoid_1205_skeleton
import os
import pickle

import numpy as np


def rebase_init_postion(pose3d, tpose):
    """
    make the pose according to T-Pose
    """
    pass


tpose = np.array(
    [
        [-0.0, 0.0, 0.92],
        [-0.1, -0.02, 0.92],
        [-0.11, -0.02, 0.49],
        [-0.14, 0.08, 0.07],
        [0.11, -0.01, 0.92],
        [0.13, 0.01, 0.49],
        [0.14, 0.1, 0.07],
        [-0.0, 0.0, 1.24],
        [-0.0, -0.03, 1.49],
        [0.0, -0.05, 1.61],
        [0.17, 0.01, 1.44],
        [0.49, 0.07, 1.42],
        [0.74, 0.09, 1.45],
        [-0.18, -0.0, 1.44],
        [-0.5, 0.04, 1.44],
        [-0.75, 0.01, 1.48],
    ]
)
print(tpose.shape)


# pose3d_f = 'estimator_inference/wild_eval/pred3D_pose/bilibili-clip/kunkun_clip_pred3D.pkl'
pose3d_f = (
    "estimator_inference/wild_eval/pred3D_pose/bilibili-clip/lumin_clip1_pred3D.pkl"
)
pose3d_world = pickle.load(open(pose3d_f, "rb"))
pose3d_world = pose3d_world["result"]
print(pose3d_world)
print(pose3d_world.shape)

# make first pose to be zero
delta = tpose - pose3d_world[1, :]
# pose3d_world += delta
print(delta)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


bvh_file = os.path.join(output_dir, "a.bvh")

Converter = humanoid_1205_skeleton.SkeletonConverter()
prediction3dpoint = Converter.convert_to_21joint(pose3d_world)
human36m_skeleton = humanoid_1205_skeleton.H36mSkeleton()
_ = human36m_skeleton.poses2bvh(prediction3dpoint, output_file=bvh_file)
