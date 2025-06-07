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

"""Base Task class."""
# 脚本的用途说明：任务基类。

import collections # 导入 collections 模块，用于创建 namedtuple（命名元组）等高级数据结构。
import os # 导入 os 模块，用于与操作系统交互，如文件路径操作。
import random # 导入 random 模块，用于生成伪随机数。
import string # 导入 string 模块，包含常用的字符串常量，如此处用于生成随机文件名。
import tempfile # 导入 tempfile 模块，用于创建临时文件和目录，如此处用于保存临时的URDF文件。

import cv2 # 导入 OpenCV (cv2) 库，一个广泛用于计算机视觉和图像处理的库。
import numpy as np # 导入 NumPy (np) 库，Python中科学计算的基础包，尤其擅长处理多维数组。
from ravens.tasks import cameras # 从 ravens.tasks 包中导入 cameras 模块。这个模块定义了仿真中使用的相机配置，例如 RealSenseD415 和 Oracle 相机。
from ravens.tasks import planners # 从 ravens.tasks 包中导入 planners 模块。这个模块定义了用于连续动作的运动规划器，例如 PickPlacePlanner。
from ravens.tasks import primitives # 从 ravens.tasks 包中导入 primitives 模块。这个模块定义了机器人可以执行的基本动作原语，如 PickPlace。
from ravens.tasks.grippers import Suction # 从 ravens.tasks.grippers 模块导入 Suction 类，它代表一个吸盘末端执行器。
# from ravens.tasks.grippers import RobotiqGripper # 这行被注释掉了，表明代码库中曾考虑或包含 Robotiq夹爪 的支持，但当前默认未使用。
from ravens.utils import utils # 从 ravens.utils 包中导入 utils 模块，包含各种通用的辅助函数，如姿态变换、颜色定义等。

import pybullet as p # 导入 pybullet 库 (别名为 p)，这是一个用于机器人和物理仿真的开源库。

class Task(): # 定义一个名为 Task 的类。所有具体的机器人任务都将继承自这个基类。
  """Base Task class.""" # 类的文档字符串，说明这是一个任务基类。

  def __init__(self, continuous = False): # Task 类的构造函数 (初始化方法)。
    """Constructor.

    Args:
      continuous: Set to `True` if you want the continuous variant.
                  (参数 continuous：布尔值，如果希望任务是连续动作空间的，则设为 True。)
    """
    self.continuous = continuous # 将传入的 continuous 参数赋值给实例变量 self.continuous，用于标记任务是否处理连续动作。
    self.ee = Suction # 设置默认的末端执行器 (end-effector, ee) 为 Suction (吸盘)。具体的任务子类可以覆盖这个属性来使用不同的末端执行器。
    self.mode = 'train' # 设置任务的默认模式为 'train'。这个模式可以影响任务的行为，例如在训练和测试时加载不同的物体集。
    self.sixdof = False # 一个布尔标志，指示任务是否为六自由度 (6DoF)。默认为 False，意味着主要考虑平面操作。MIRA等6DoF任务会将其设为True。
    if continuous: # 如果任务是连续动作空间的 (self.continuous 为 True)。
      self.primitive = primitives.PickPlaceContinuous() # 将动作原语 (self.primitive) 设置为连续的 PickPlace 原语。
    else: # 如果任务是离散动作空间的。
      self.primitive = primitives.PickPlace() # 将动作原语设置为离散的 PickPlace 原语。
    self.oracle_cams = cameras.Oracle.CONFIG # 设置专家策略 (oracle) 使用的相机配置。通常是 cameras.Oracle.CONFIG，这是一个完美的顶视正交相机配置，用于专家获取场景的无噪声、完整信息。

    # Evaluation epsilons (for pose evaluation metric).
    # 评估误差容忍度（用于姿态评估指标）。
    self.pos_eps = 0.01 # 位置误差容忍度，通常单位是米 (0.01 米 = 1 厘米)。用于判断一个物体是否到达了目标位置。
    self.rot_eps = np.deg2rad(15) # 旋转误差容忍度，将15度转换为弧度。用于判断物体的姿态是否与目标姿态足够接近。

    # Whether we need to match roll and pitch.
    # 标志位，指示在姿态匹配时是否需要同时匹配翻滚角 (roll) 和俯仰角 (pitch)。
    self.match_rp = False # 默认为 False。如果为 True，is_match 函数会更严格地检查旋转。

    # Workspace bounds.
    # 定义工作空间的边界。
    self.pix_size = 0.003125 # 图像中每个像素代表的物理尺寸（米/像素）。这个值用于在像素坐标和世界坐标之间进行转换。
    self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]]) # 定义工作空间在 x, y, z 轴上的最小和最大边界。
                                                                # 格式为 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]。机器人通常在这个范围内操作。

    self.goals = [] # 初始化一个空列表，用于存储任务的目标。具体的任务子类在其 reset 方法中会填充这个列表。
    self.progress = 0 # 初始化任务进展度量为0，范围通常在 [0, 1] 之间。
    self._rewards = 0 # 初始化内部变量 _rewards 为0，用于累计已返回的奖励，以计算增量奖励。

    self.assets_root = None # 初始化资源文件根目录的路径为 None。这个路径需要在任务使用前通过 set_assets_root() 方法进行设置。

  def reset(self, env):  # pylint: disable=unused-argument (env 参数在基类中未直接使用，但子类在调用 super().reset(env) 时会传入)
    # 当环境重置时调用的方法，用于设置任务的初始状态和目标。
    if not self.assets_root: # 检查 self.assets_root 是否已经被设置。
      raise ValueError('assets_root must be set for task, ' # 如果未设置，则抛出 ValueError 异常。
                       'call set_assets_root().')
    self.goals = [] # 清空目标列表，为新的任务回合准备。
    self.progress = 0  # 将任务进展重置为0。
    self._rewards = 0  # 将累计奖励重置为0。
    if self.continuous: # 如果任务是连续动作空间的。
      self.primitive.reset() # 调用动作原语的 reset 方法 (如果该原语有 reset 方法的话，例如 PickPlaceContinuous)。

  #-------------------------------------------------------------------------
  # Oracle Agent (专家智能体部分)
  #-------------------------------------------------------------------------

  def oracle(self, env, **kwargs): # 定义获取专家智能体的方法。
    """Oracle agent.""" # 文档字符串。
    # **kwargs 允许传递如 steps_per_seg 这样的额外参数。
    if self.continuous: # 如果任务是连续动作的。
      return self._continuous_oracle(env, **kwargs) # 返回一个为连续动作设计的专家智能体。
    return self._discrete_oracle(env) # 否则，返回一个为离散动作设计的专家智能体。

  def _continuous_oracle(self, env, **kwargs): # 定义获取连续动作专家智能体的方法 (受保护成员)。
    """Continuous oracle agent.

    This oracle will generate the pick and place poses using the original
    discrete oracle. It will then interpolate intermediate actions using
    splines.
    (这个专家会使用原始的离散专家生成抓取和放置姿态，然后使用样条插值生成中间动作。)
    Args:
      env: The environment instance. (环境实例)
      **kwargs: extra kwargs for the oracle. (其他额外参数)
    Returns:
      ContinuousOracle. (返回 ContinuousOracle 类的实例)
    """
    kwargs['env'] = env # 将环境实例添加到关键字参数字典中。
    kwargs['base_oracle_cls'] = self._discrete_oracle # 将离散专家 (_discrete_oracle 方法本身) 作为参数传递。
    kwargs['ee'] = self.ee # 将当前任务的末端执行器类型作为参数传递。
    return ContinuousOracle(**kwargs) # 创建并返回 ContinuousOracle 类的实例 (ContinuousOracle 类定义在文件的更后面)。

  def _discrete_oracle(self, env): # 定义获取离散动作专家智能体的方法 (受保护成员)。
    """Discrete oracle agent."""
    OracleAgent = collections.namedtuple('OracleAgent', ['act']) # 使用命名元组 collections.namedtuple 创建一个简单的 Agent 结构，
                                                               # 它只有一个名为 'act' 的字段 (将来会赋值为一个函数)。

    def act(obs, info):  # pylint: disable=unused-argument (obs 和 info 在这个通用 oracle 中不直接使用所有部分，因为它能直接访问真实状态)
      """Calculate action.""" # 定义实际的动作计算函数 act。
                              # 这个函数将作为 OracleAgent 实例的 act 方法。

      # Oracle uses perfect RGB-D orthographic images and segmentation masks.
      # (专家策略使用完美的RGB-D正交图像和分割掩码)
      _, hmap, obj_mask = self.get_true_image(env) # 调用 self.get_true_image 方法获取场景的“真实”高度图(hmap)和物体分割掩码(obj_mask)。
                                                 # _ 通常是彩色图，这里未使用。

      # Unpack next goal step.
      # (解包下一个目标步骤)
      # self.goals 是一个列表，存储了任务需要完成的各个目标。每个目标是一个元组。
      # 这里获取列表中的第一个目标 self.goals[0]。
      objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]
      # objs: 一个列表，包含要操作的物体及其对称性信息。格式: [(body_id, (symmetry_value, None)), ...]
      # matches: 一个 NumPy 数组，表示物体与目标姿态之间的匹配关系。
      # targs: 一个列表，包含目标姿态。格式: [(position_xyz, orientation_xyzw), ...]
      # replace: 布尔值，指示物体在匹配后是否应被视为“已处理”或“已消耗”。
      # rotations: 布尔值，指示在放置时是否需要考虑物体的旋转。
      # 后面的 '_' 是占位符，对应 self.goals 元组中定义的 metric, params, max_reward，在动作决策中不直接使用。

      # Match objects to targets without replacement.
      # (不放回地将物体与目标匹配)
      if not replace: # 如果 replace 标志为 False (即物体匹配后不从考虑中移除，适用于需要同时满足多个物体目标的情况)。
        # Modify a copy of the match matrix.
        # (修改匹配矩阵的副本)
        matches = matches.copy() # 创建 matches 矩阵的副本，以避免修改 self.goals 中的原始数据。

        # Ignore already matched objects.
        # (忽略已经匹配的物体)
        for i in range(len(objs)): # 遍历所有待操作的物体。
          object_id, (symmetry, _) = objs[i] # 获取物体的 PyBullet ID 和对称性信息。
          pose = p.getBasePositionAndOrientation(object_id) # 获取物体当前的真实姿态 (位置和方向四元数)。
          targets_i = np.argwhere(matches[i, :]).reshape(-1) # 从匹配矩阵中找出当前物体 i 还可以匹配的目标姿态的索引。
          for j in targets_i: # 遍历这些可能的目标姿态索引 j。
            if self.is_match(pose, targs[j], symmetry): # 调用 self.is_match 方法检查物体当前姿态是否已经满足目标姿态 targs[j] (在误差容限内)。
              matches[i, :] = 0 # 如果已匹配，则将匹配矩阵中物体 i 对应的行全部置零，表示该物体不再需要移动。
              matches[:, j] = 0 # 同时将目标姿态 j 对应的列全部置零，表示该目标已被满足。
              break # 跳出内层循环，因为物体 i 已匹配。

      # Get objects to be picked (prioritize farthest from nearest neighbor).
      # (获取要拾取的物体（策略是优先选择离其最近的未满足目标最远的那个物体）)
      nn_dists = [] # 存储每个物体到其最近的、尚未满足的目标位置的距离。
      nn_targets = [] # 存储每个物体对应的那个最近的、尚未满足的目标的索引。
      for i in range(len(objs)): # 再次遍历所有待操作的物体。
        object_id, (symmetry, _) = objs[i] # 获取物体ID和对称性。
        xyz, _ = p.getBasePositionAndOrientation(object_id) # 获取物体当前的位置。
        targets_i = np.argwhere(matches[i, :]).reshape(-1) # 从（可能已更新的）匹配矩阵中找出当前物体 i 还可以匹配的目标姿态的索引。
        if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test (如果该物体还有未满足的目标)。
          targets_xyz = np.float32([targs[j][0] for j in targets_i]) # 获取这些未满足目标的位置。
          dists = np.linalg.norm(
              targets_xyz - np.float32(xyz).reshape(1, 3), axis=1) # 计算物体当前位置到每个未满足目标位置的欧氏距离。
          nn = np.argmin(dists) # 找到距离最近的那个目标的索引 (在 targets_i 内部的索引)。
          nn_dists.append(dists[nn]) # 将这个最小距离添加到 nn_dists 列表。
          nn_targets.append(targets_i[nn]) # 将这个最近目标的原始索引 (在 targs 列表中的索引) 添加到 nn_targets 列表。
        else: # 如果该物体已经没有未满足的目标了。
          nn_dists.append(0) # 距离设为0。
          nn_targets.append(-1) # 目标索引设为-1（无效值）。
      order = np.argsort(nn_dists)[::-1] # 对所有物体的 nn_dists (到最近目标的距离) 进行降序排序，得到物体索引的顺序。
                                       # 这样，距离其最近目标最远的物体会排在最前面，被优先考虑抓取。

      # Filter out matched objects.
      # (再次过滤掉已经匹配的物体，即那些 nn_dists[i] 为 0 的物体)
      order = [i for i in order if nn_dists[i] > 0]

      pick_mask = None # 初始化抓取候选物体的像素掩码为 None。
      for pick_i in order: # 按照上面计算出的优先级顺序 (order) 遍历物体。
        # 从 obj_mask (通过 get_true_image 获取的完美分割图) 中提取出当前物体 objs[pick_i][0] 的像素掩码。
        pick_mask = np.uint8(obj_mask == objs[pick_i][0])

        # Erode to avoid picking on edges.
        # (可选) 腐蚀掩码，避免在物体边缘进行抓取，可以使抓取更稳定。
        pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

        if np.sum(pick_mask) > 0: # 如果这个物体的掩码不为空 (即物体在Oracle相机视野中可见)。
          break # 找到了要抓取的物体，跳出循环。

      # Trigger task reset if no object is visible.
      # (如果遍历完所有优先级的物体后，仍然没有找到可见的、可抓取的物体)
      if pick_mask is None or np.sum(pick_mask) == 0: # (pick_mask 仍然是 None 或者掩码全为0)
        self.goals = [] # 清空任务目标列表 (意味着这个演示回合失败或无法进行)。
        print('Object for pick is not visible. Skipping demonstration.') # 打印提示信息。
        return # 返回 None，表示专家无法提供有效动作。

      # Get picking pose.
      # (计算抓取姿态)
      pick_prob = np.float32(pick_mask) # 将选定物体的像素掩码转换为浮点型，可以看作是在物体表面均匀采样的概率分布。
      pick_pix = utils.sample_distribution(pick_prob) # 从这个概率分布中采样一个像素点 (y, x) 作为抓取点。
      # For "deterministic" demonstrations on insertion-easy, use this:
      # (对于一些简单的、需要确定性演示的任务，可以直接指定抓取像素)
      # pick_pix = (160,80)
      pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                  self.bounds, self.pix_size) # 使用工具函数将2D像素坐标 pick_pix 和高度图 hmap 转换为3D世界坐标 pick_pos。

      # 计算抓取姿态：根据末端执行器类型选择不同的姿态计算方法
      if self.ee.__name__ == 'RobotiqGripper':
        # 对于夹爪，计算智能抓取姿态和位置
        pick_orientation = self._compute_gripper_grasp_orientation(objs[pick_i][0], pick_pos)  # pylint: disable=undefined-loop-variable
        # 调整夹爪的抓取高度，使其更接近物体
        pick_pos = self._adjust_gripper_pick_position(objs[pick_i][0], pick_pos)  # pylint: disable=undefined-loop-variable
      else:
        # 对于吸盘，使用默认的单位四元数（向下抓取）
        pick_orientation = np.asarray((0, 0, 0, 1))

      pick_pose = (np.asarray(pick_pos), pick_orientation)

      # Get placing pose.
      # (计算放置姿态)
      # nn_targets[pick_i] 是之前为选定抓取物体 (pick_i) 找到的最近的目标姿态的索引。
      targ_pose = targs[nn_targets[pick_i]]  # pylint: disable=undefined-loop-variable (获取该物体的目标放置姿态)
      obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])  # pylint: disable=undefined-loop-variable (获取该物体当前的真实姿态)

      # 如果任务不是六自由度 (self.sixdof 为 False)，则简化物体和目标姿态，通常只保留偏航角(yaw)。
      if not self.sixdof:
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1]) # 获取物体当前姿态的欧拉角。
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2])) # 创建一个新的四元数，只包含偏航角信息。
        obj_pose = (obj_pose[0], obj_quat) # 更新物体姿态。

        targ_euler = utils.quatXYZW_to_eulerXYZ(targ_pose[1]) # 获取目标姿态的欧拉角。
        targ_quat = utils.eulerXYZ_to_quatXYZW((0, 0, targ_euler[2])) # 创建一个新的四元数，只包含偏航角信息。
        targ_pose = (targ_pose[0], targ_quat) # 更新目标姿态。

      # 计算末端执行器放置姿态的核心逻辑:
      # 目的是使得当末端执行器到达 place_pose 并释放物体时，物体能够精确地处于 targ_pose。
      world_to_pick = utils.invert(pick_pose) # 计算从世界坐标系到抓取点 pick_pose 的变换矩阵的逆，即从抓取点到世界坐标系的变换。
                                            # (这里应该是从世界到抓取点的变换，utils.invert 如果输入是 T_world_pick，输出是 T_pick_world)
                                            # 更准确地说，pick_pose 是 T_world_pickTip (末端执行器在抓取时的世界姿态)
                                            # world_to_pick 应该是 T_pickTip_world
      obj_to_pick = utils.multiply(world_to_pick, obj_pose) # 计算物体姿态 obj_pose 在抓取点坐标系下的表示。
                                                            # T_pickTip_world * T_world_object = T_pickTip_object
                                                            # 这代表了抓取点相对于物体原点的变换。
      pick_to_obj = utils.invert(obj_to_pick) # 计算 obj_to_pick 的逆变换，即 T_object_pickTip。
                                               # 这代表了从物体原点到抓取点的变换。
      place_pose = utils.multiply(targ_pose, pick_to_obj) # 计算最终的放置姿态。
                                                          # T_world_targetObject * T_object_pickTip = T_world_placeTip
                                                          # 当末端执行器处于 place_pose 时，如果它以相对于物体相同的抓取方式持有物体，
                                                          # 那么物体就会被放置在 targ_pose。

      # Rotate end effector?
      # (是否需要旋转末端执行器进行放置？)
      if not rotations: # rotations 参数来自 self.goals[0]，由具体任务在其 reset 方法中定义。
        place_pose = (place_pose[0], (0, 0, 0, 1)) # 如果不需要考虑放置时的旋转，则将放置姿态的旋转部分设为单位四元数。

      # 将位置和姿态转换为 NumPy 数组。
      place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

      # 返回一个包含抓取姿态 ('pose0') 和放置姿态 ('pose1') 的字典作为专家动作。
      return {'pose0': pick_pose, 'pose1': place_pose}

    return OracleAgent(act) # 将定义的 act 函数包装成 OracleAgent 对象并返回。

  #-------------------------------------------------------------------------
  # Reward Function and Task Completion Metrics
  #-------------------------------------------------------------------------

  def reward(self):
    """Get delta rewards for current timestep.

    Returns:
      A tuple consisting of the scalar (delta) reward, plus `extras`
        dict which has extra task-dependent info from the process of
        computing rewards that gives us finer-grained details. Use
        `extras` for further data analysis.
    """
    reward, info = 0, {}

    if self.goals:
      # Unpack next goal step.
      objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]

      # Evaluate by matching object poses.
      if metric == 'pose':
        step_reward = 0
        for i in range(len(objs)):
          object_id, (symmetry, _) = objs[i]
          pose = p.getBasePositionAndOrientation(object_id)
          targets_i = np.argwhere(matches[i, :]).reshape(-1)
          for j in targets_i:
            target_pose = targs[j]
            if self.is_match(pose, target_pose, symmetry):
              step_reward += max_reward / len(objs)
              break

      # Evaluate by measuring object intersection with zone.
      elif metric == 'zone':
        zone_pts, total_pts = 0, 0
        obj_pts, zones = params
        for zone_pose, zone_size in zones:

          # Count valid points in zone.
          for obj_id in obj_pts:
            pts = obj_pts[obj_id]
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            world_to_zone = utils.invert(zone_pose)
            obj_to_zone = utils.multiply(world_to_zone, obj_pose)
            pts = np.float32(utils.apply(obj_to_zone, pts))
            if len(zone_size) > 1:
              valid_pts = np.logical_and.reduce([
                  pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
                  pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
                  pts[2, :] < self.bounds[2, 1]])

            zone_pts += np.sum(np.float32(valid_pts))
            total_pts += pts.shape[1]
        step_reward = max_reward * (zone_pts / total_pts)

      # Get cumulative rewards and return delta.
      reward = self.progress + step_reward - self._rewards
      self._rewards = self.progress + step_reward

      # Move to next goal step if current goal step is complete.
      if np.abs(max_reward - step_reward) < 0.01:
        self.progress += max_reward  # Update task progress.
        self.goals.pop(0)

    else:
      # At this point we are done with the task but executing the last movements
      # in the plan. We should return 0 reward to prevent the total reward from
      # exceeding 1.0.
      reward = 0.0

    return reward, info

  def done(self):
    """Check if the task is done or has failed.

    Returns:
      True if the episode should be considered a success, which we
        use for measuring successes, which is particularly helpful for tasks
        where one may get successes on the very last time step, e.g., getting
        the cloth coverage threshold on the last alllowed action.
        However, for bag-items-easy and bag-items-hard (which use the
        'bag-items' metric), it may be necessary to filter out demos that did
        not attain sufficiently high reward in external code. Currently, this
        is done in `main.py` and its ignore_this_demo() method.
    """

    # # For tasks with self.metric == 'pose'.
    # if hasattr(self, 'goal'):
    # goal_done = len(self.goal['steps']) == 0  # pylint:
    # disable=g-explicit-length-test
    return (len(self.goals) == 0) or (self._rewards > 0.99)  # pylint: disable=g-explicit-length-test
    # return zone_done or defs_done or goal_done

  #-------------------------------------------------------------------------
  # Environment Helper Functions
  #-------------------------------------------------------------------------

  def is_match(self, pose0, pose1, symmetry):
    """Check if pose0 and pose1 match within a threshold."""

    # Get translational error.
    diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
    dist_pos = np.linalg.norm(diff_pos)

    # Get rotational error around z-axis (account for symmetries).
    diff_rot = 0
    rot_is_close = (diff_rot < self.rot_eps)
    
    if symmetry > 0:
      rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
      rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
      diff_rot = np.abs(rot0 - rot1) % symmetry
      if diff_rot > (symmetry / 2):
        diff_rot = symmetry - diff_rot
      rot_is_close = (diff_rot < self.rot_eps)
    pos_is_close = (dist_pos < self.pos_eps)

    if self.match_rp:
      # Yaw
      if symmetry > 0:
        rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
        rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
        diff_rot_y = np.abs(rot0 - rot1) % symmetry
        if diff_rot_y > (symmetry / 2):
          diff_rot_y = symmetry - diff_rot
      else:
        diff_rot_y = 0

      # Roll, Pitch
      rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[:2]
      rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[:2]
      diff_rot_rp = np.abs(rot0 - rot1)
      
      # Increase `self.pos_eps` and `self.rot_eps` because 
      # hanging-disks-ood shows that disk could hit the floor 
      # and affect the positions.
      # TODO: Change the task to fix this.
      diff_rot = (diff_rot_rp[0], diff_rot_rp[1], diff_rot_y)
      rot_is_close = (diff_rot < self.rot_eps*1.5).all()
      pos_is_close = (dist_pos < self.pos_eps*2)

    
    return pos_is_close and rot_is_close

  def get_true_image(self, env):
    """Get RGB-D orthographic heightmaps and segmentation masks."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(self.oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = utils.reconstruct_heightmaps(
        [color], [depth], self.oracle_cams, self.bounds, self.pix_size)

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
    return cmap, hmap, mask

  def get_random_pose(self, env, obj_size):
    """Get random collision-free object pose within workspace bounds."""

    # Get erosion size of object in pixels.
    max_size = np.sqrt(obj_size[0]**2 + obj_size[1]**2)
    erode_size = int(np.round(max_size / self.pix_size))

    _, hmap, obj_mask = self.get_true_image(env)

    # Randomly sample an object pose within free-space pixels.
    free = np.ones(obj_mask.shape, dtype=np.uint8)
    for obj_ids in env.obj_ids.values():
      for obj_id in obj_ids:
        free[obj_mask == obj_id] = 0
    free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
    free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
    if np.sum(free) == 0:
      return None, None
    pix = utils.sample_distribution(np.float32(free))
    pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
    pos = (pos[0], pos[1], obj_size[2] / 2)
    theta = np.random.rand() * 2 * np.pi
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    return pos, rot

  #-------------------------------------------------------------------------
  # Helper Functions
  #-------------------------------------------------------------------------

  def fill_template(self, template, replace):
    """Read a file and replace key strings."""
    full_template_path = os.path.join(self.assets_root, template)
    with open(full_template_path, 'r') as file:
      fdata = file.read()
    for field in replace:
      for i in range(len(replace[field])):
        fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
    alphabet = string.ascii_lowercase + string.digits
    rname = ''.join(random.choices(alphabet, k=16))
    tmpdir = tempfile.gettempdir()
    template_filename = os.path.split(template)[-1]
    fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
    with open(fname, 'w') as file:
      file.write(fdata)
    return fname

  def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
    """Get random box size."""
    size = np.random.rand(3)
    size[0] = size[0] * (max_x - min_x) + min_x
    size[1] = size[1] * (max_y - min_y) + min_y
    size[2] = size[2] * (max_z - min_z) + min_z
    return tuple(size)

  def get_box_object_points(self, obj):
    obj_shape = p.getVisualShapeData(obj)
    obj_dim = obj_shape[0][3]
    obj_dim = tuple(d for d in obj_dim)
    xv, yv, zv = np.meshgrid(
      np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
      np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
      np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
      sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def get_mesh_object_points(self, obj):
    mesh = p.getMeshData(obj)
    mesh_points = np.array(mesh[1])
    mesh_dim = np.vstack((mesh_points.min(axis=0), mesh_points.max(axis=0)))
    xv, yv, zv = np.meshgrid(
          np.arange(mesh_dim[0][0], mesh_dim[1][0], 0.02),
          np.arange(mesh_dim[0][1], mesh_dim[1][1], 0.02),
          np.arange(mesh_dim[0][2], mesh_dim[1][2], 0.02),
          sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def get_object_points(self, obj):
    obj_shape = p.getVisualShapeData(obj)
    obj_dim = obj_shape[0][3]
    xv, yv, zv = np.meshgrid(
        np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
        np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
        np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
        sparse=False, indexing='xy')
    return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

  def color_random_brown(self, obj):
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    p.changeVisualShape(obj, -1, rgbaColor=color)

  def set_assets_root(self, assets_root):
    self.assets_root = assets_root

  def _compute_gripper_grasp_orientation(self, obj_id, pick_pos):
    """计算夹爪的智能抓取姿态。

    Args:
      obj_id: PyBullet物体ID
      pick_pos: 抓取位置 (x, y, z)

    Returns:
      四元数表示的抓取姿态 (x, y, z, w)
    """
    try:
      # 获取物体的当前姿态和几何信息
      obj_pos, obj_quat = p.getBasePositionAndOrientation(obj_id)

      # 获取物体的视觉形状数据来估计尺寸
      visual_shape_data = p.getVisualShapeData(obj_id)
      if visual_shape_data:
        # 获取物体的边界框尺寸
        obj_dimensions = visual_shape_data[0][3]  # (length, width, height)

        # 计算物体的长宽比，选择最优的抓取方向
        length, width, height = obj_dimensions

        # 获取物体当前的偏航角
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_quat)
        obj_yaw = obj_euler[2]

        # 策略1：如果物体是长方形，沿着较短边抓取
        if abs(length - width) > 0.01:  # 不是正方形
          if length > width:
            # 物体较长，夹爪应该垂直于长边（沿着宽边方向）
            grasp_yaw = obj_yaw + np.pi/2
          else:
            # 物体较宽，夹爪应该垂直于宽边（沿着长边方向）
            grasp_yaw = obj_yaw
        else:
          # 正方形物体，可以选择任意方向，这里选择与物体对齐
          grasp_yaw = obj_yaw

        # 策略2：添加一些随机性以提高鲁棒性
        # 在最优角度基础上添加小的随机偏移（±15度）
        random_offset = (np.random.rand() - 0.5) * np.pi/6  # ±30度
        grasp_yaw += random_offset

        # 策略3：考虑多个候选角度，选择最佳的
        candidate_yaws = [
          grasp_yaw,
          grasp_yaw + np.pi/2,
          obj_yaw,
          obj_yaw + np.pi/4,
          obj_yaw + np.pi/2,
          obj_yaw + 3*np.pi/4
        ]

        # 简单启发式：选择与物体主轴最对齐的角度
        # 这里我们选择第一个候选角度作为基础实现
        final_yaw = candidate_yaws[0]

        # 确保角度在合理范围内
        final_yaw = final_yaw % (2 * np.pi)

        # 创建抓取姿态：保持夹爪垂直向下，但调整偏航角
        # roll=0, pitch=0（垂直向下），yaw=计算得出的角度
        grasp_orientation = utils.eulerXYZ_to_quatXYZW((0, 0, final_yaw))

        return np.asarray(grasp_orientation)

    except Exception as e:
      # 如果计算失败，回退到默认姿态
      print(f"Warning: Failed to compute gripper orientation for object {obj_id}: {e}")
      return np.asarray((0, 0, 0, 1))

    # 默认情况：返回单位四元数
    return np.asarray((0, 0, 0, 1))

  def _adjust_gripper_pick_position(self, obj_id, pick_pos):
    """调整夹爪的抓取位置，使其更适合夹爪操作。

    Args:
      obj_id: PyBullet物体ID
      pick_pos: 原始抓取位置 (x, y, z)

    Returns:
      调整后的抓取位置 (x, y, z)
    """
    try:
      # 获取物体的当前位置和几何信息
      obj_pos, _ = p.getBasePositionAndOrientation(obj_id)

      # 获取物体的视觉形状数据来估计尺寸
      visual_shape_data = p.getVisualShapeData(obj_id)
      if visual_shape_data:
        obj_dimensions = visual_shape_data[0][3]  # (length, width, height)
        obj_height = obj_dimensions[2]

        # 计算合适的抓取高度：物体顶部 + 小的偏移
        # 这样夹爪可以从上方接近物体
        grasp_height = obj_pos[2] + obj_height/2 + 0.01  # 物体顶部 + 1cm偏移

        # 保持x, y坐标不变，只调整z坐标
        adjusted_pos = (pick_pos[0], pick_pos[1], grasp_height)

        return adjusted_pos

    except Exception as e:
      print(f"Warning: Failed to adjust gripper position for object {obj_id}: {e}")
      return pick_pos

    # 默认情况：返回原始位置
    return pick_pos


class ContinuousOracle:
  """Continuous oracle."""

  def __init__(
      self,
      env,
      base_oracle_cls,
      ee,
      t_max = 10.0,
      steps_per_seg = 2,
      height = 0.32,
  ):
    """Constructor.

    Args:
      env:
      base_oracle_cls:
      ee:
      t_max:
      steps_per_seg:
      height:
    """
    self._env = env
    self._base_oracle = base_oracle_cls(env)
    self.steps_per_seg = steps_per_seg

    planner_cls = planners.PickPlacePlanner if ee == Suction else planners.PushPlanner
    self._planner = planner_cls(steps_per_seg, t_max, height)

    self._actions = []

  def act(self, obs, info):
    """Get oracle action from planner."""
    if not self._actions:
      # Query the base oracle for pick and place poses.
      act = self._base_oracle.act(obs, info)
      if act is None:
        return
      self._actions = self._planner(self._env.get_ee_pose(), act['pose0'],
                                    act['pose1'])
    act = self._actions.pop(0)
    act['acts_left'] = len(self._actions)
    return act

  @property
  def num_poses(self):
    return self._planner.NUM_POSES
