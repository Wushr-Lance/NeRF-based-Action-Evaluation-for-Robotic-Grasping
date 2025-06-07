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

"""Sorting Task."""
# 脚本的用途说明：分类任务。

import numpy as np # 导入 numpy 用于数值计算。
from ravens.tasks.task import Task # 从 ravens.tasks.task 模块导入 Task 基类，所有具体任务都继承自它。
# from ravens.tasks.grippers import RobotiqGripper # 这行被注释掉了，表明原始代码可能考虑过 Robotiq 夹爪，但当前版本未使用。
from ravens.utils import utils # 导入 ravens.utils.utils 模块，包含一些通用辅助函数，如颜色定义、姿态变换等。

import pybullet as p # 导入 pybullet，物理仿真引擎。

# 定义基础任务类 PlaceRedInGreen
class PlaceRedInGreen(Task):
  """Sorting Task."""
  # 类的文档字符串：分类任务。

  def __init__(self, *args, **kwargs):
    # 构造函数。*args 和 **kwargs 允许传递任意数量的位置参数和关键字参数给父类的构造函数。
    super().__init__(*args, **kwargs) # 调用父类 Task 的构造函数。
    self.max_steps = 10 # 设置此任务允许的最大执行步数为10。如果超过这个步数任务仍未完成，通常认为失败。
    self.pos_eps = 0.05 # 位置误差容忍度，单位可能是米。用于判断物体是否被放置到了目标位置附近。
    # self.ee = RobotiqGripper # 这行被注释掉了，进一步表明默认情况下不使用 RobotiqGripper。Task 基类中 self.ee 默认为 Suction (吸盘)。

  def reset(self, env):
    # 当环境重置时调用的方法，用于设置任务的初始场景。
    # env 是一个 Environment 实例。
    super().reset(env) # 调用父类 Task 的 reset 方法，会清空 self.goals 列表等。
    n_bowls = np.random.randint(1, 4) # 随机生成碗的数量，范围在 [1, 3] 个。
    n_blocks = np.random.randint(1, n_bowls + 1) # 随机生成红色物块的数量，数量少于等于碗的数量。

    # Add bowls.
    # 添加碗到场景中。
    bowl_size = (0.12, 0.12, 0) # 定义碗的尺寸 (x, y, z)，z为0意味着它是一个扁平的碗底。
    bowl_urdf = 'bowl/bowl.urdf' # 碗的 URDF 文件路径 (相对于 assets_root)。
    bowl_poses = [] # 用于存储所有碗的姿态。
    for _ in range(n_bowls): # 循环创建指定数量的碗。
      bowl_pose = self.get_random_pose(env, bowl_size) # 调用 Task 基类中的 get_random_pose 方法，在工作空间内为碗生成一个随机的、无碰撞的姿态。
      env.add_object(bowl_urdf, bowl_pose, 'fixed') # 将碗作为固定物体 (fixed) 添加到仿真环境中。
      bowl_poses.append(bowl_pose) # 将碗的姿态保存起来，后续用于定义目标。

    # Add blocks.
    # 添加（红色的）物块到场景中。
    blocks = [] # 用于存储所有物块的 ID 和对称性信息。
    block_size = (0.04, 0.04, 0.04) # 定义物块的尺寸（立方体）。
    block_urdf = 'stacking/block.urdf' # 物块的 URDF 文件路径。
    for _ in range(n_blocks): # 循环创建指定数量的物块。
      block_pose = self.get_random_pose(env, block_size) # 为物块生成一个随机的、无碰撞的姿态。
      block_id = env.add_object(block_urdf, block_pose) # 将物块作为可动物体添加到仿真环境中，并获取其 PyBullet ID。
      blocks.append((block_id, (0, None))) # 将物块 ID 和对称性信息 (0 表示360度旋转对称或不考虑对称性，None 是对称性的额外参数) 添加到列表中。
                                            # 这些物块默认是红色的，因为这是 "PlaceRedInGreen" 任务。颜色是在加载URDF后通过 p.changeVisualShape 设置的，或者URDF本身定义了颜色。
                                            # 在这个基础类中，物块颜色依赖于URDF默认或由通用着色逻辑处理。目标是红色方块放入绿色碗。

    # Goal: each red block is in a different green bowl.
    # 目标：每个红色物块都在一个不同的绿色碗中。
    # 这行代码是定义专家演示目标的核心。
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))
    # 参数解释 (对应 Task 基类中 oracle 读取的结构):
    # - blocks: 要操作的物体列表，每个元素是 (obj_id, (symmetry, params))。
    # - np.ones((len(blocks), len(bowl_poses))): 匹配矩阵。这里是一个全1矩阵，表示任何一个红色物块可以匹配任何一个绿色碗的姿态（专家策略会确保一一对应）。
    # - bowl_poses: 目标姿态列表，即所有绿色碗的姿态。
    # - False (replace): 物体被放置后是否从待处理列表中移除。False 表示不移除，适用于多目标匹配。
    # - True (rotations): 放置时是否考虑旋转。
    # - 'pose' (metric): 评估任务成功的标准是基于物体姿态是否与目标姿态匹配。
    # - None (params): 特定评估指标所需的额外参数。
    # - 1 (max_reward): 完成此目标的最高奖励。

    # Colors of distractor objects.
    # 定义干扰物体的颜色。
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green'] # 干扰碗的颜色：除了绿色以外的所有预定义颜色。
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red'] # 干扰物块的颜色：除了红色以外的所有预定义颜色。
                                                                      # utils.COLORS 在 ravens/utils/utils.py 中定义。

    # Add distractors.
    # 添加干扰物体到场景中。
    n_distractors = 0 # 初始化干扰物体计数器。
    while n_distractors < 10: # 循环直到添加了10个干扰物体。
      is_block = np.random.rand() > 0.5 # 随机决定当前要添加的是物块还是碗。
      urdf = block_urdf if is_block else bowl_urdf # 根据随机结果选择对应的 URDF 文件。
      size = block_size if is_block else bowl_size # 根据随机结果选择对应的尺寸。
      colors = block_colors if is_block else bowl_colors # 根据随机结果选择对应的颜色列表。
      pose = self.get_random_pose(env, size) # 为干扰物体生成随机姿态。
      if not pose[0] or not pose[1]: # 如果生成的姿态无效（例如，找不到无碰撞位置），则跳过此次添加。
        continue
      obj_id = env.add_object(urdf, pose) # 将干扰物体添加到环境中。
      color = colors[n_distractors % len(colors)] # 从相应的颜色列表中循环选择一个颜色。
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1]) # 使用 PyBullet 函数改变物体的视觉颜色。颜色列表末尾添加的 [1] 是 alpha 通道（不透明）。
      n_distractors += 1 # 干扰物体计数器加一。

# 定义 PlaceRedInGreenSixDofDiscrete 类，继承自 PlaceRedInGreen
class PlaceRedInGreenSixDofDiscrete(PlaceRedInGreen):
  """Placing Task - 6DOF Variant."""
  # 类的文档字符串：放置任务 - 六自由度变种。

  # Class variables.
  # 定义类变量，用于离散化的旋转角度。
  rolls = [-np.pi/6, 0, np.pi/6] # 绕x轴的翻滚角选项。
  pitchs = [-np.pi/6, 0, np.pi/6] # 绕y轴的俯仰角选项。

  @classmethod
  def get_rolls(cls): # 类方法，返回允许的翻滚角列表。
      return cls.rolls

  @classmethod
  def get_pitchs(cls): # 类方法，返回允许的俯仰角列表。
      return cls.pitchs

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs) # 调用父类 PlaceRedInGreen 的构造函数。
    self.sixdof = True # 设置标志，表示这个任务是六自由度的。Task 基类可能会使用这个标志来调整专家策略的行为。

  def reset(self, env):
    super(PlaceRedInGreen, self).reset(env) # 调用 PlaceRedInGreen 的 reset 方法，而不是 Task 基类的。
                                           # 这里会执行 PlaceRedInGreen 的 reset 逻辑，但我们下面会覆盖掉大部分。
                                           # 实际上，这里更常见的写法是 super().__init__(*args, **kwargs) 之后，再写自己的 reset 逻辑。
                                           # 或者，如果希望完全重写，则不调用 super().reset()，或者只调用最顶层 Task.reset()。
                                           # 不过，由于 PlaceRedInGreen 的 reset 主要是添加物体和目标，这里的 super 调用方式问题不大，但后续的 goals.append 会再次添加。
                                           # 更佳实践是只调用 Task.reset() 或在 super() 调用后 carefully 地修改 self.goals。
                                           # 鉴于后续代码完全重新定义了物体和目标，这里的 super(PlaceRedInGreen, self).reset(env) 之后，之前的目标会被新的覆盖。
    n_bowls = np.random.randint(1, 4) # 重新定义碗的数量。
    # n_blocks = np.random.randint(1, n_bowls + 1) # 原始的多物块逻辑被注释掉。
    n_blocks = 1 # 在这个六自由度版本中，只使用1个红色物块。

    # Add bowls.
    # 添加碗，与基础类类似，但碗的姿态是六自由度的。
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'bowl/bowl.urdf'
    bowl_poses = []
    for _ in range(n_bowls):
      bowl_pose = self.get_random_pose_6dof(env, bowl_size) # 调用下面定义的 get_random_pose_6dof 方法。
      env.add_object(bowl_urdf, bowl_pose, 'fixed')
      bowl_poses.append(bowl_pose)

    # Add blocks.
    # 添加物块，与基础类类似。
    blocks = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    for _ in range(n_blocks): # 只循环一次，因为 n_blocks = 1。
      block_pose = self.get_random_pose(env, block_size) # 物块的初始姿态是普通的随机姿态（可能主要是平面）。
      block_id = env.add_object(block_urdf, block_pose)
      blocks.append((block_id, (0, None)))

    # Goal: each red block is in a different green bowl.
    # 定义目标，与基础类结构相同。
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))

    # Colors of distractor objects.
    # 添加干扰物体的逻辑与基础类完全相同。
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

    # Add distractors.
    n_distractors = 0
    while n_distractors < 10:
      # ... (与基础类相同的干扰物体添加逻辑)
      is_block = np.random.rand() > 0.5
      urdf = block_urdf if is_block else bowl_urdf
      size = block_size if is_block else bowl_size
      colors = block_colors if is_block else bowl_colors
      pose = self.get_random_pose(env, size)
      if not pose[0] or not pose[1]:
        continue
      obj_id = env.add_object(urdf, pose)
      color = colors[n_distractors % len(colors)]
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
      n_distractors += 1

  def get_random_pose_6dof(self, env, obj_size):
    # 为物体（这里是碗）生成一个随机的六自由度姿态。
    pos, rot = self.get_random_pose(env, obj_size) # 首先获取一个基础的随机平面姿态 (主要x,y和yaw随机)。
    z = 0.03 # 设置一个固定的z轴高度偏移。
    pos = (pos[0], pos[1], obj_size[2] + z) # 更新位置的z值。
    pitch = np.random.choice(self.pitchs) # 从预定义的离散俯仰角列表中随机选择一个。
    roll = np.random.choice(self.rolls) # 从预定义的离散翻滚角列表中随机选择一个。
    yaw = np.random.rand() * 2 * np.pi # 随机生成一个偏航角 (0 到 2*pi)。
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw)) # 将欧拉角 (roll, pitch, yaw) 转换为四元数。
    return pos, rot # 返回生成的六自由度姿态 (位置, 四元数姿态)。

# 定义 PlaceRedInGreenSixDof 类，继承自 PlaceRedInGreen
class PlaceRedInGreenSixDof(PlaceRedInGreen):
  """Placing Task - 6DOF Variant."""
  # 与 PlaceRedInGreenSixDofDiscrete 类似，但 roll 和 pitch 是在连续范围内随机生成的。

  # Class variables.
  roll_bounds = (-np.pi/6, np.pi/6) # 定义翻滚角的边界范围。
  pitch_bounds = (-np.pi/6, np.pi/6) # 定义俯仰角的边界范围。
  n_rotations = 7 # 用于 NeRF 视图生成的旋转数量（可能与 MIRA 的某些配置相关，但不直接影响这里的姿态生成）。
  rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist() # 生成一组离散的翻滚角（主要供 get_rolls 使用）。
  pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist() # 生成一组离散的俯仰角（主要供 get_pitchs 使用）。

  @classmethod
  def get_rolls(cls): # 类方法，返回生成的翻滚角列表。
      return cls.rolls

  @classmethod
  def get_pitchs(cls): # 类方法，返回生成的俯仰角列表。
      return cls.pitchs

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs) # 调用父类 PlaceRedInGreen 的构造函数。
    self.sixdof = True # 设置为六自由度任务。

  def reset(self, env): # reset 方法的实现与 PlaceRedInGreenSixDofDiscrete 中的 reset 非常相似。
    super(PlaceRedInGreen, self).reset(env) # 调用 PlaceRedInGreen 的 reset。
    n_bowls = np.random.randint(1, 4)
    n_blocks = 1

    # Add bowls.
    bowl_size = (0.12, 0.12, 0)
    bowl_urdf = 'bowl/bowl.urdf'
    bowl_poses = []
    for _ in range(n_bowls):
      bowl_pose = self.get_random_pose_6dof(env, bowl_size) # 调用此类中定义的 get_random_pose_6dof。
      env.add_object(bowl_urdf, bowl_pose, 'fixed')
      bowl_poses.append(bowl_pose)

    # Add blocks.
    blocks = []
    block_size = (0.04, 0.04, 0.04)
    block_urdf = 'stacking/block.urdf'
    for _ in range(n_blocks):
      block_pose = self.get_random_pose(env, block_size)
      block_id = env.add_object(block_urdf, block_pose)
      blocks.append((block_id, (0, None)))

    # Goal: each red block is in a different green bowl.
    self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                       bowl_poses, False, True, 'pose', None, 1))

    # Colors of distractor objects.
    bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
    block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']

    # Add distractors.
    # ... (与基础类相同的干扰物体添加逻辑)
    n_distractors = 0
    while n_distractors < 10:
      is_block = np.random.rand() > 0.5
      urdf = block_urdf if is_block else bowl_urdf
      size = block_size if is_block else bowl_size
      colors = block_colors if is_block else bowl_colors
      pose = self.get_random_pose(env, size)
      if not pose[0] or not pose[1]:
        continue
      obj_id = env.add_object(urdf, pose)
      color = colors[n_distractors % len(colors)]
      p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
      n_distractors += 1

  def get_random_pose_6dof(self, env, obj_size):
    # 为物体（碗）生成一个随机的六自由度姿态，roll 和 pitch 在连续范围内取值。
    pos, rot = super(PlaceRedInGreenSixDof, self).get_random_pose(env, obj_size) # 调用 PlaceRedInGreen 的 get_random_pose，获取基础平面姿态。
                                                                            # 注意：这里 super() 的用法可能需要斟酌，通常会用 super() 直接调用父类的方法，
                                                                            # 或者 super(ClassName, self).method_name()。
                                                                            # Task 基类中已经有 get_random_pose。
    z = 0.03
    pos = (pos[0], pos[1], obj_size[2] / 2 + z) # 设置 z 值。obj_size[2]/2 是为了让物体底部接触平面。
    roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0] # 在定义的边界内随机生成翻滚角。
    pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0] # 在定义的边界内随机生成俯仰角。
    yaw = np.random.rand() * 2 * np.pi # 随机生成偏航角。
    rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw)) # 转换为四元数。
    return pos, rot

# 定义 PlaceRedInGreenSixDofOOD 类，继承自 PlaceRedInGreenSixDof
class PlaceRedInGreenSixDofOOD(PlaceRedInGreenSixDof):
    "Hanging Disks Out of Distribution" # 文档字符串似乎拷贝自其他任务，应为 "Place Red in Green Out of Distribution"。
                                       # OOD = Out of Distribution，表示这个任务变种用于测试模型在训练分布之外的数据上的泛化能力。

    # Class variables.
    # 重新定义了更大的 roll 和 pitch 边界，以及更多的离散旋转点（用于 get_rolls/get_pitchs，但 get_random_pose_6dof 中未使用这些离散点）。
    roll_bounds = (-np.pi/4, np.pi/4) # 更大的翻滚角范围。
    pitch_bounds = (-np.pi/4, np.pi/4) # 更大的俯仰角范围。
    n_rotations = 11
    rolls = np.linspace(roll_bounds[0], roll_bounds[1], n_rotations).tolist()
    pitchs = np.linspace(pitch_bounds[0], pitch_bounds[1], n_rotations).tolist()

    @classmethod
    def get_rolls(cls):
        return cls.rolls

    @classmethod
    def get_pitchs(cls):
        return cls.pitchs

    def get_random_pose_6dof(self, env, obj_size):
        # 为物体（碗）生成一个六自由度姿态，其中 roll 或 pitch 会被强制设置为分布外的角度。
        pos, rot = self.get_random_pose(env, obj_size) # 获取基础平面姿态。
        z = 0.03
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        # 首先在原始（训练分布内）的 roll_bounds 和 pitch_bounds 生成 roll 和 pitch。
        # 注意：这里使用的是 self.roll_bounds 和 self.pitch_bounds，它们已经被此类覆盖为 (-np.pi/4, np.pi/4)。
        # 但下面的 OOD 逻辑会进一步修改它们。
        roll = np.random.rand() * (self.roll_bounds[1]-self.roll_bounds[0]) + self.roll_bounds[0]
        pitch = np.random.rand() * (self.pitch_bounds[1]-self.pitch_bounds[0]) + self.pitch_bounds[0]

        ood = np.random.choice(['roll', 'pitch']) # 随机选择是 roll 还是 pitch 作为 OOD 维度。
        if ood == 'roll':
            # 生成一个在 np.pi/6 到 self.roll_bounds[1] (即 np.pi/4) 之间（或其负区间）的 roll 值。
            # 这确保了 roll 值会落在标准 PlaceRedInGreenSixDof 任务的 roll_bounds (-np.pi/6, np.pi/6) 之外。
            roll = np.random.rand() * (self.roll_bounds[1]-np.pi/6) + np.pi/6
            roll = -1 * roll if np.random.rand() > 0.5 else roll 
        elif ood == 'pitch':
            # 类似地，生成一个分布外的 pitch 值。
            pitch = np.random.rand() * (self.pitch_bounds[1]-np.pi/6) + np.pi/6
            pitch = -1 * pitch if np.random.rand() > 0.5 else pitch
            
        yaw = np.random.rand() * 2 * np.pi # 偏航角依然随机。
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw)) # 转换为四元数。
        return pos, rot