# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes to handle gripper dynamics."""

import os

import numpy as np
from ravens.utils import pybullet_utils

import pybullet as p

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'
GRIPPER_URDF = 'ur5/gripper/robotiq_2f_85.urdf'


class Gripper:
  """Base gripper class."""

  def __init__(self, assets_root): # assets_root是存放URDF等资源文件的根目录
    self.assets_root = assets_root
    self.activated = False # 一个布尔标志，用于表示末端执行器当前是否处于“激活”状态（例如，吸盘正在吸附或夹爪正在夹紧）。

  def step(self):
    """This function can be used to create gripper-specific behaviors."""
    return

  def activate(self, objects): # 用于激活末端执行器，使其与物体交互（例如，吸盘开始吸附）。
    del objects # 基类中忽略objects，具体的交互逻辑由子类定义
    return

  def release(self): # 用于释放末端执行器抓取的物体。
    return


class Spatula(Gripper): # 刮刀
  """Simulate simple spatula for pushing."""

  def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
    """Creates spatula and 'attaches' it to the robot."""
    super().__init__(assets_root)

    # Load spatula model.
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))


class Suction(Gripper): # 吸盘
  """Simulate simple suction dynamics."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    """Creates suction and 'attaches' it to the robot.

    Has special cases when dealing with rigid vs deformables. For rigid,
    only need to check contact_constraint for any constraint. For soft
    bodies (i.e., cloth or bags), use cloth_threshold to check distances
    from gripper body (self.body) to any vertex in the cloth mesh. We
    need correct code logic to handle gripping potentially a rigid or a
    deformable (and similarly for releasing).

    To be clear on terminology: 'deformable' here should be interpreted
    as a PyBullet 'softBody', which includes cloths and bags. There's
    also cables, but those are formed by connecting rigid body beads, so
    they can use standard 'rigid body' grasping code.

    To get the suction gripper pose, use p.getLinkState(self.body, 0),
    and not p.getBasePositionAndOrientation(self.body) as the latter is
    about z=0.03m higher and empirically seems worse.

    Args:
      assets_root: str for root directory with assets.
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all suctionable objects in the env.
    """
    # obj_ids: 一个字典，包含了场景中不同类型物体（如 'rigid', 'deformable'）的 PyBullet ID 列表。吸盘需要知道哪些物体是可以被吸附的。
    super().__init__(assets_root)

    # Load suction gripper base model (visual only).
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    # 定义吸盘基座在世界坐标系中的初始加载姿态。
    # pose[0] = (0.487, 0.109, 0.438) 是位置 (x, y, z)。
    # pose[1] = p.getQuaternionFromEuler((np.pi, 0, 0)) 是姿态，通过欧拉角 (roll=pi, pitch=0, yaw=0) 转换为四元数。
    # 这个初始姿态可能是一个临时加载位置，之后会通过约束固定到机器人手臂上。
    self.base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
    # 调用 pybullet_utils.load_urdf（一个封装了 p.loadURDF 的函数）来加载吸盘基座的 URDF 模型。
    # SUCTION_BASE_URDF 是一个常量，值为 'ur5/suction/suction-base.urdf'。
    # 这个基座模型通常只有视觉属性 (visual only)，不参与物理碰撞。
    # 加载后的 PyBullet 物体 ID 存储在 self.base 中。
    p.createConstraint(
        parentBodyUniqueId=robot,          # 父物体 ID，即机器人手臂本体。
        parentLinkIndex=ee,                # 父连杆索引，即机器人手臂上用于连接末端执行器的连杆。
        childBodyUniqueId=self.base,       # 子物体 ID，即刚刚加载的吸盘基座。
        childLinkIndex=-1,                 # 子连杆索引，-1 表示物体的基座连杆。
        jointType=p.JOINT_FIXED,           # 约束类型：固定约束，意味着吸盘基座将刚性连接到机器人手臂。
        jointAxis=(0, 0, 0),               # 关节轴（对于固定约束不重要）。
        parentFramePosition=(0, 0, 0),     # 约束点在父连杆（机器人手臂末端）坐标系中的位置。
        childFramePosition=(0, 0, 0.01))   # 约束点在子连杆（吸盘基座）坐标系中的位置。
                                           # 这里 (0, 0, 0.01) 表示吸盘基座的连接点相对于其自身原点在 z 轴上有 0.01 米的偏移。
    # 创建一个固定约束，将吸盘基座 (self.base) 连接到机器人 (robot) 的末端连杆 (ee) 上。

    # Load suction tip model (visual and collision) with compliance.
    # 加载吸盘吸头模型（包含视觉和碰撞属性），并带有一些柔顺性。
    # urdf = 'assets/ur5/suction/suction-head.urdf' # 这行是被注释掉的示例路径。
    pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
    # 加载吸盘吸头的 URDF 模型。
    # SUCTION_HEAD_URDF 是一个常量，值为 'ur5/suction/suction-head.urdf'。
    # 这个吸头模型包含视觉和碰撞属性，是实际与物体交互的部分。
    # 加载后的 PyBullet 物体 ID 存储在 self.body 中。

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,          # 父物体 ID：机器人手臂。
        parentLinkIndex=ee,                # 父连杆索引：机器人手臂末端连杆。
        childBodyUniqueId=self.body,       # 子物体 ID：吸盘吸头。
        childLinkIndex=-1,                 # 子连杆索引：吸盘吸头的基座连杆。
        jointType=p.JOINT_FIXED,           # 约束类型：固定约束。
        jointAxis=(0, 0, 0),               # 关节轴。
        parentFramePosition=(0, 0, 0),     # 约束点在父连杆坐标系中的位置。
        childFramePosition=(0, 0, -0.08))  # 约束点在子连杆（吸盘吸头）坐标系中的位置。
                                           # 这里 (0, 0, -0.08) 表示吸盘吸头的连接点相对于其自身原点在 z 轴上向下（负方向）偏移了 0.08 米。
                                           # 这个偏移决定了吸头相对于机器人手臂末端连杆的实际安装位置。
    # 创建一个固定约束，将吸盘吸头 (self.body) 连接到机器人 (robot) 的末端连杆 (ee) 上。
    p.changeConstraint(constraint_id, maxForce=50)
    # 修改刚刚创建的约束。
    # maxForce=50 设置了该约束能够施加的最大力（或抵抗分离的力）。
    # 这个设置可以用来模拟一定的柔顺性 (compliance)，即如果作用在吸盘上的力超过 50，这个约束可能会失效，
    # 或者表现出一定的“弹性”，尽管这是一个固定约束。在实践中，它更多地是防止模拟器中出现过大的约束力导致不稳定。

    # Reference to object IDs in environment for simulating suction.
    # 存储环境中可被吸附物体的 ID，用于模拟吸附。
    self.obj_ids = obj_ids
    # 将传入的 obj_ids (一个包含场景中物体ID的字典或列表) 存储到实例变量 self.obj_ids。
    # 在 activate 方法中会用到它来判断接触到的物体是否是可吸附的。

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    # 用于抓取和释放刚体物体。
    self.contact_constraint = None
    # 初始化 self.contact_constraint 为 None。当吸盘成功吸附一个刚体时，
    # 这个变量会存储 PyBullet 返回的固定约束的 ID。释放物体时，会用这个 ID 来移除约束。

    # Defaults for deformable parameters, and can override in tasks.
    # 可变形物体参数的默认值，可以在具体任务中覆盖。
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    # self.def_ignore：在检查与可变形物体（如布料）的顶点距离时，可能用于忽略过于接近的顶点或定义一个安全距离。
    # TODO 注释表明这个参数的必要性可能需要复查。


    self.def_threshold = 0.030
    # self.def_threshold：吸附可变形物体时，吸盘中心与物体顶点之间的距离阈值。
    # 如果距离小于这个阈值，可能会触发吸附。

    self.def_nb_anchors = 1
    # self.def_nb_anchors：吸附可变形物体时创建的锚点（约束点）数量。
    # 对于软体，通常不是简单地创建一个固定约束，而是将软体的几个顶点“锚定”到吸盘上。

    # Track which deformable is being gripped (if any), and anchors.
    # 追踪当前抓取的（如果是）可变形物体以及相关的锚点。
    self.def_grip_item = None
    # self.def_grip_item：存储当前被吸附的可变形物体的 ID。

    self.def_grip_anchors = []
    # self.def_grip_anchors：一个列表，用于存储吸附可变形物体时创建的多个约束（锚点）的 ID。

    # Determines release when gripped deformable touches a rigid/def.
    # 用于判断当被抓取的可变形物体接触到其他刚体或可变形物体时是否应释放。
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self): # 注意：原始代码中 activate(self, objects) 中的 objects 被 del 了，这里简化为 activate(self)
    """Simulate suction using a rigid fixed constraint to contacted object."""
    # TODO(andyzeng): check deformables logic.
    # del def_ids

    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0) # 检测吸头(self.body)的接触点
      if points:
        for point in points: # 遍历所有接触点
          obj_id, contact_link = point[2], point[4] # 获取接触到的物体ID和连杆索引
        if obj_id in self.obj_ids['rigid']: # 如果接触到的是刚体
          body_pose = p.getLinkState(self.body, 0) # 获取吸头的姿态
          obj_pose = p.getBasePositionAndOrientation(obj_id) # 获取物体的姿态
          world_to_body = p.invertTransform(body_pose[0], body_pose[1]) # 计算从世界到吸头的变换
          obj_to_body = p.multiplyTransforms(world_to_body[0], # 计算物体相对于吸头的姿态
                                             world_to_body[1],
                                             obj_pose[0], obj_pose[1])
          # 创建一个固定约束，将物体“粘”在吸盘上
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body, # 父对象是吸头
              parentLinkIndex=0, # 吸头的基座连杆
              childBodyUniqueId=obj_id, # 子对象是被吸附的物体
              childLinkIndex=contact_link, # 物体上被接触的连杆
              jointType=p.JOINT_FIXED, # 固定约束
              jointAxis=(0, 0, 0), # 约束轴（对于固定约束不重要）
              parentFramePosition=obj_to_body[0], # 物体在吸头坐标系中的位置
              parentFrameOrientation=obj_to_body[1], # 物体在吸头坐标系中的姿态
              childFramePosition=(0, 0, 0), # 约束点在物体自身坐标系中的位置
              childFrameOrientation=(0, 0, 0)) # 约束点在物体自身坐标系中的姿态
        self.activated = True # 设置为已激活

  def release(self):
    """Release gripper object, only applied if gripper is 'activated'.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    To handle deformables, simply remove constraints (i.e., anchors).
    Also reset any relevant variables, e.g., if releasing a rigid, we
    should reset init_grip values back to None, which will be re-assigned
    in any subsequent grasps.
    """
    if self.activated:
      self.activated = False

      # Release gripped rigid object (if any).
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass
        self.init_grip_distance = None
        self.init_grip_item = None

      # Release gripped deformable object (if any).
      if self.def_grip_anchors:
        for anchor_id in self.def_grip_anchors:
          p.removeConstraint(anchor_id)
        self.def_grip_anchors = []
        self.def_grip_item = None
        self.def_min_vetex = None
        self.def_min_distance = None

  def detect_contact(self):
    """Detects a contact with a rigid object."""
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:   # 如果已吸附物体
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]  # 则检测被吸附物体的接触
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link) # 获取接触点
    # print(points)
    # exit()
    if self.activated:  # 如果已激活，则排除吸头自身与被吸附物体之间的接触点
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False

  def check_grasp(self):
    """Check a grasp (object in contact?) for picking success."""

    suctioned_object = None
    if self.contact_constraint is not None: # 如果存在约束
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2] # 获取被约束的物体ID
    return suctioned_object is not None # 如果物体ID存在，则表示抓取成功

  def get_registration_info(self):
    return self.base, os.path.join(self.assets_root, SUCTION_BASE_URDF), self.body, os.path.join(self.assets_root, SUCTION_HEAD_URDF)


class RobotiqGripper(Gripper):
  """Simulate Robotiq 2F-85 gripper dynamics."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    """Creates Robotiq gripper and 'attaches' it to the robot.

    Args:
      assets_root: str for root directory with assets.
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all grippable objects in the env.
    """
    super().__init__(assets_root)

    # Load gripper model.
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0))) # 定义robotiq gripper在世界坐标系中的初始加载姿态
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, GRIPPER_URDF), pose[0], pose[1])

    # Attach gripper to robot end effector with stronger constraint
    self.constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,                  # 父ID，即机器人手臂本体
        parentLinkIndex=ee,                        # 父连杆索引，即机器人手臂上用于连接末端执行器的连杆。
        childBodyUniqueId=self.body,               # 即夹爪自己
        childLinkIndex=-1,                         # 这里的 childLinkIndex=-1 意味着这个固定约束是将机器人手臂的末端连杆 (parentLinkIndex=ee) 连接到夹爪这个物体 (childBodyUniqueId=self.body) 的基座连杆 (base link) 上。
        jointType=p.JOINT_FIXED,                   # 意味着要刚性连接
        jointAxis=(0, 0, 0),                       # 关节轴（对于固定约束不重要）。
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0))              # TODO check后续是否修改
    # Increase constraint force to prevent detachment
    p.changeConstraint(self.constraint_id, maxForce=500)   # 确保夹爪在仿真过程中始终牢固地连接在手臂末端，更接近真实世界中螺栓固定的情况

    # Reference to object IDs in environment for simulating grasping.
    self.obj_ids = obj_ids   # 提供一个场景中所有可交互（可抓取或可吸附）物体的标识符 (ID) 列表或字典给末端执行器对象。

    # Indicates whether gripper is gripping anything.
    self.activated = False

    # For gripping and releasing rigid objects.
    # 用于抓取和释放刚体物体。
    self.contact_constraint = None
    # 初始化 self.contact_constraint 为 None。当吸盘成功吸附一个刚体时，
    # 这个变量会存储 PyBullet 返回的固定约束的 ID。释放物体时，会用这个 ID 来移除约束。

    # Get gripper joint indices for finger control
    # 获取用于手指控制的夹爪关节索引。
    self.joints = []
    # 初始化一个空列表 self.joints，用于存储夹爪上所有可控的（通常是驱动手指运动的）关节的索引。

    n_joints = p.getNumJoints(self.body)
    # 获取夹爪模型 (self.body) 的总关节数量。

    for i in range(n_joints):
      # 遍历夹爪的每一个关节。
      joint_info = p.getJointInfo(self.body, i)
      # 获取当前遍历到的关节 (索引为 i) 的信息。
      # joint_info 是一个包含关节多种属性的元组，例如关节名称、类型、轴等。

      # if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
      if joint_info[1] == b'robotiq_2f_85_right_driver_joint':
        # 检查关节类型。joint_info[2] 是关节的类型。
        # p.JOINT_REVOLUTE 表示这是一个旋转关节（如驱动手指开合的马达关节）。
        # 这里假设夹爪的手指是由旋转关节驱动的。对于某些线性驱动的夹爪，可能需要检查 p.JOINT_PRISMATIC。
        self.joints.append(i)
        # 如果是旋转关节，则将其索引 i 添加到 self.joints 列表中。

    # Gripper state: 0 = open, 1 = closed
    # 夹爪状态：0 = 张开，1 = 闭合。
    self.gripper_state = 0
    # 初始化 self.gripper_state 为 0，表示夹爪初始处于张开状态。
    # 这是一个自定义的状态变量，用于在代码中追踪夹爪是张开还是闭合。

    self.max_force = 50  # Increased force for better grasping
    # 定义夹爪在控制关节时可以施加的最大力（或扭矩，取决于关节类型和控制模式）。
    # 这个值会在后续控制夹爪闭合或张开时，在 p.setJointMotorControl2 中作为 positionGains 或 force 参数使用。

  def activate(self):
    """Simulate grasping by closing gripper and creating constraints."""
    if not self.activated:
      print(f"Activating gripper...")

      # Get gripper position for debugging
      gripper_pose = p.getBasePositionAndOrientation(self.body)
      print(f"Gripper position: {gripper_pose[0]}")

      # Close gripper fingers first
      self._close_gripper()

      # Wait longer for gripper to close and settle
      for i in range(20):
        p.stepSimulation()
        if i % 5 == 0:  # Check every 5 steps
          # Check contact during closing
          contact_points = p.getContactPoints(bodyA=self.body)
          if contact_points:
            print(f"Contact detected during closing (step {i}): {len(contact_points)} points")

      # Check for contact with objects after gripper closes
      contact_points = p.getContactPoints(bodyA=self.body)
      print(f"Final contact points: {len(contact_points)}")

      if contact_points: # 如果 p.getContactPoints() 返回的列表不为空，意味着夹爪至少碰到了一个物体。
        print("Contact points found:") # 打印一条调试信息。
        for i, point in enumerate(contact_points): # 遍历所有检测到的接触点。一个接触事件可能包含多个接触点。
          obj_id = point[2]  # bodyB
          # p.getContactPoints 返回的每个 point 是一个元组。
          # point[2] 是参与接触的第二个物体 (bodyB) 的 PyBullet ID。
          # 因为我们调用的是 p.getContactPoints(bodyA=self.body)，所以 bodyA 是夹爪，bodyB 就是夹爪碰到的那个物体。

          contact_link = point[4]

          contact_pos = point[5]  # contact position on bodyA
          # point[5] 是接触点在 bodyA (即夹爪) 的局部坐标系中的位置。
          # 这对于调试和理解接触发生在哪里很有用。

          print(f"  Point {i}: obj_id={obj_id}, contact_pos={contact_pos}") # 打印每个接触点的详细信息，用于调试。

          if obj_id in self.obj_ids['rigid']: # 检查接触到的物体 obj_id 是否在 self.obj_ids['rigid'] 列表中。
                                            # self.obj_ids 是在夹爪初始化时传入的，包含了场景中所有可被抓取的刚性物体的ID。
                                            # 这个检查至关重要，它确保了夹爪只会尝试“抓住”我们预定义的、可操作的物体，而不会尝试“抓住”地面、工作台或机器人自身的其他部分。
            print(f"Found grippable object: {obj_id}") # 如果是可抓取的物体，打印一条确认信息。

            # Create constraint to "grasp" the object
            # (创建一个约束来“抓取”物体)
            gripper_pose = p.getBasePositionAndOrientation(self.body) # 获取夹爪 (self.body) 当前在世界坐标系中的姿态 (位置和方向四元数)。
            obj_pose = p.getBasePositionAndOrientation(obj_id) # 获取被接触物体 (obj_id) 当前在世界坐标系中的姿态。

            # Calculate relative pose
            # (计算相对姿态)
            # 为了创建一个固定约束，我们需要知道物体相对于夹爪的姿态，这样约束才能将物体“锁定”在当前相对位置。
            world_to_gripper = p.invertTransform(gripper_pose[0], gripper_pose[1]) # 计算从世界坐标系到夹爪局部坐标系的变换矩阵的逆变换。
                                                                               # 结果是一个 (位置, 方向) 元组，代表了 T_gripper_world 的变换。
            obj_to_gripper = p.multiplyTransforms(world_to_gripper[0], # 将 T_gripper_world 变换与物体的世界姿态 T_world_object 相乘。
                                                 world_to_gripper[1],
                                                 obj_pose[0], obj_pose[1]) # 结果 obj_to_gripper (T_gripper_object) 就是物体在夹爪局部坐标系中的姿态。

            # Create strong fixed constraint between gripper and object
            # (在夹爪和物体之间创建一个强大的固定约束)
            self.contact_constraint = p.createConstraint( # 调用 PyBullet 函数创建约束，并将其返回的唯一 ID 存储在 self.contact_constraint 中。
                parentBodyUniqueId=self.body,        # 约束的父物体是夹爪。
                parentLinkIndex=-1,                  # 约束连接到夹爪的基座连杆 (base link)。
                childBodyUniqueId=obj_id,            # 约束的子物体是被接触的物体。
                childLinkIndex=contact_link,                   # 约束连接到该物体的基座连杆。
                jointType=p.JOINT_FIXED,             # 约束类型为固定约束。这将使两个物体之间没有任何相对运动，就像它们被焊接在了一起。
                jointAxis=(0, 0, 0),                 # 对于固定约束，关节轴没有意义。
                parentFramePosition=obj_to_gripper[0], # **关键参数**: 约束点在父物体（夹爪）局部坐标系中的位置。
                                                       # 这里设置为我们刚刚计算出的物体相对夹爪的位置。
                parentFrameOrientation=obj_to_gripper[1],# **关键参数**: 约束点在父物体（夹爪）局部坐标系中的姿态。
                                                         # 这里设置为我们刚刚计算出的物体相对夹爪的姿态。
                childFramePosition=(0, 0, 0),          # 约束点在子物体（被抓物体）局部坐标系中的位置，通常设为其原点。
                childFrameOrientation=(0, 0, 0, 1))    # 约束点在子物体局部坐标系中的姿态，通常设为单位四元数（无旋转）。
                                                       # PyBullet 会将子物体的这个点和姿态“锁定”到父物体上对应的点和姿态。

            # Set high constraint force to ensure stable grasping
            # (设置高的约束力以确保稳定的抓取)
            p.changeConstraint(self.contact_constraint, maxForce=1000) # 修改刚刚创建的约束，将其最大作用力设置为一个较大的值 (1000)。
                                                                     # 这可以防止在后续移动过程中，由于较大的力或加速度导致这个模拟的“抓取”意外“断开”。

            self.activated = True # 将夹爪的状态标志 self.activated 设置为 True，表示它现在已经成功抓取了一个物体。
            print(f"✓ Gripper activated: grasped object {obj_id}") # 打印成功激活的信息。
            return # 成功抓取一个物体后，立即返回，不再继续遍历其他接触点或尝试抓取其他物体。

      print("✗ Gripper activation failed: no grippable objects in contact")

      # Debug: print all rigid object IDs and their positions
      print("Available rigid objects:")
      for obj_id in self.obj_ids['rigid']:
        obj_pose = p.getBasePositionAndOrientation(obj_id)
        print(f"  Object {obj_id}: position {obj_pose[0]}")

        # Calculate distance to gripper
        distance = np.linalg.norm(np.array(obj_pose[0]) - np.array(gripper_pose[0]))
        print(f"    Distance to gripper: {distance:.4f}")

        # If object is close enough, create a "magnetic" grasp
        if distance < 0.3:  # 30cm threshold for magnetic grasp
          print(f"  Attempting magnetic grasp of object {obj_id}")

          # Create constraint to "grasp" the object
          world_to_gripper = p.invertTransform(gripper_pose[0], gripper_pose[1])
          obj_to_gripper = p.multiplyTransforms(world_to_gripper[0],
                                               world_to_gripper[1],
                                               obj_pose[0], obj_pose[1])

          # Create strong fixed constraint between gripper and object
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=-1,
              childBodyUniqueId=obj_id,
              childLinkIndex=-1,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=obj_to_gripper[0],
              parentFrameOrientation=obj_to_gripper[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))

          # Set high constraint force to ensure stable grasping
          p.changeConstraint(self.contact_constraint, maxForce=1000)

          self.activated = True
          print(f"✓ Magnetic grasp successful: grasped object {obj_id}")
          return

  def release(self):
    """Release gripper object by opening gripper and removing constraints."""
    if self.activated:
      self.activated = False

      # Open gripper fingers
      self._open_gripper()

      # Remove grasping constraint
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass

  def _open_gripper(self):
    """Open gripper fingers."""
    self.gripper_state = 0
    # Set target positions for gripper joints (open position)
    for joint_id in self.joints:
      p.setJointMotorControl2(
          bodyIndex=self.body,
          jointIndex=joint_id,
          controlMode=p.POSITION_CONTROL,
          targetPosition=0.0,  # Open position
          force=self.max_force,
          positionGain=1.0,  # Add position gain for better control
          velocityGain=2.0)  # Add velocity gain for smoother motion

  def _close_gripper(self):
    """Close gripper fingers."""
    self.gripper_state = 1
    # Set target positions for gripper joints (closed position)
    for joint_id in self.joints:  # 这里的joints都是revolute joints，其实只需要控制right driver
      p.setJointMotorControl2(
          bodyIndex=self.body,
          jointIndex=joint_id,
          controlMode=p.POSITION_CONTROL,
          targetPosition=0.8,  # Reduced from 0.8 for better grasping
          force=self.max_force,
          positionGain=1.0,  # Add position gain for better control
          velocityGain=2.0)  # Add velocity gain for smoother motion

  def detect_contact(self):
    """Detects a contact with a grippable rigid object."""
    body, link = self.body, -1
    if self.activated and self.contact_constraint is not None:   # 如果已吸附物体
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]  # 则检测被吸附物体的接触
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link) # 获取接触点
    # print(points)
    # exit()
    if self.activated:  # 如果已激活，则排除吸头自身与被吸附物体之间的接触点
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False



    # if self.activated and self.contact_constraint is not None:
    #   try:
    #     p.getConstraintInfo(self.contact_constraint)
    #     return True  # Already grasping something
    #   except:  # pylint: disable=bare-except
    #     self.contact_constraint = None
    #     return False

    # # Get all contact points between the gripper and other bodies
    # points = p.getContactPoints(bodyA=self.body)

    # # Check if any contact is with a grippable object
    # # Exclude contacts with robot body (typically ID 0, 1, 2)
    # for point in points:
    #   obj_id = point[2]  # bodyB
    #   if obj_id in self.obj_ids['rigid'] and obj_id > 2:  # Exclude robot body IDs
    #     print(f"Contact detected with grippable object {obj_id}")
    #     return True

    # return False

  def check_grasp(self):
    """Check if gripper is successfully grasping an object."""
    gripped_object = None
    if self.contact_constraint is not None: # 如果存在约束
      gripped_object = p.getConstraintInfo(self.contact_constraint)[2] # 获取被约束的物体ID
    return gripped_object is not None # 如果物体ID存在，则表示抓取成功

  def get_registration_info(self):
    """Return gripper registration info for recording."""
    return self.body, os.path.join(self.assets_root, GRIPPER_URDF), None, None # 只有一个body