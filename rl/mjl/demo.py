import mujoco


# XML_MODEL_PATH = 'models/humanoid_h36m/humanoid_h36m_v5.xml'
XML_MODEL_PATH = 'models/mug/mug.xml'
# XML_MODEL_PATH = '/home/jintian/mujoco/model/humanoid/humanoid.xml'
model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
data = mujoco.MjData(model)

while data.time < 1:
    mujoco.mj_step(model, data)
    print(data.geom_xpos)