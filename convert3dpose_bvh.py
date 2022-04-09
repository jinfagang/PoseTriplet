from video2bvh.bvh_skeleton import h36m_skeleton, cmu_skeleton
from video2bvh.bvh_skeleton import humanoid_1205_skeleton
from video2bvh.utils import vis
import os
import pickle
import numpy as np



pose3d_f = 'estimator_inference/wild_eval/pred3D_pose/bilibili-clip/kunkun_clip_pred3D.pkl'
pose3d_world = pickle.load(open(pose3d_f, 'rb'))
pose3d_world = pose3d_world['result']
print(pose3d_world)
print(pose3d_world.shape)

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
h36m_skel = h36m_skeleton.H36mSkeleton()
gif_file = os.path.join(output_dir, '3d_pose_300.gif')

# ani = vis.vis_3d_keypoints_sequence(
#     keypoints_sequence=pose3d_world[0:300],
#     skeleton=h36m_skel,
#     azimuth=np.array(70., dtype=np.float32),
#     fps=60,
#     output_file=gif_file
# )
# ani.to_html5_video()

bvh_file = os.path.join(output_dir, 'a.bvh')

Converter = humanoid_1205_skeleton.SkeletonConverter()
prediction3dpoint = Converter.convert_to_21joint(pose3d_world)

human36m_skeleton = humanoid_1205_skeleton.H36mSkeleton()
_ = h36m_skel.poses2bvh(prediction3dpoint, output_file=bvh_file)