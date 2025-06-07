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

"""Data collection script."""
# 脚本的用途说明：数据收集脚本。

import os # 导入 os 模块，用于与操作系统交互，例如文件路径操作。

from absl import app # 从 absl库导入 app，用于创建命令行应用。
from absl import flags # 从 absl库导入 flags，用于定义和解析命令行参数。

import numpy as np # 导入 numpy，用于数值计算，特别是数组操作。
import json # 导入 json，用于处理 JSON 数据格式（可能用于配置文件或元数据）。
import pybullet as p # 导入 pybullet，物理仿真引擎。

from ravens import tasks # 导入 ravens.tasks 模块，其中定义了各种机器人任务。
from ravens.dataset import Dataset # 从 ravens.dataset 模块导入 Dataset 类，用于管理和存储收集到的数据。
from ravens.environments.environment import Environment # 从 ravens.environments.environment 模块导入 Environment 类，用于创建和管理仿真环境。
from ravens.utils.demo_utils import write_nerf_data # 从 ravens.utils.demo_utils 导入 write_nerf_data 函数，用于为NeRF准备数据。

from PIL import Image # 导入 PIL (Pillow) 库中的 Image模块，通常用于图像处理（这里没有直接使用，但可能是某些依赖库需要）。


# 定义命令行参数
flags.DEFINE_string('assets_root', '.', '') # 定义 'assets_root' 参数，字符串类型，默认值为 '.' (当前目录)，用于指定资源文件根目录。
flags.DEFINE_string('data_dir', '.', '') # 定义 'data_dir' 参数，字符串类型，默认值为 '.'，用于指定保存数据集的目录。
flags.DEFINE_bool('disp', False, '') # 定义 'disp' 参数，布尔类型，默认值为 False，控制是否显示PyBullet的GUI界面。
flags.DEFINE_bool('shared_memory', False, '') # 定义 'shared_memory' 参数，布尔类型，默认值为 False，控制是否使用共享内存（通常与GUI一起使用）。
flags.DEFINE_string('task', 'towers-of-hanoi', '') # 定义 'task' 参数，字符串类型，默认值为 'towers-of-hanoi'，指定要执行的任务名称。
flags.DEFINE_string('mode', 'train', '') # 定义 'mode' 参数，字符串类型，默认值为 'train'，指定数据收集的模式（如 'train' 或 'test'）。
flags.DEFINE_integer('n', 1000, '') # 定义 'n' 参数，整数类型，默认值为 1000，指定要收集的演示数据回合数。
flags.DEFINE_bool('continuous', False, '') # 定义 'continuous' 参数，布尔类型，默认值为 False，指定任务是否是连续动作空间。
flags.DEFINE_integer('steps_per_seg', 3, '') # 定义 'steps_per_seg' 参数，整数类型，默认值为 3，用于连续动作任务中每个动作段的步数。
flags.DEFINE_integer('n_input_views', 36, '') # 定义 'n_input_views' 参数，整数类型，默认值为 36，指定用于NeRF训练的输入视角的数量。
flags.DEFINE_bool('debug', False, '') # 定义 'debug' 参数，布尔类型，默认值为 False，如果为True，则可能跳过某些耗时操作（如数据存储或NeRF训练）。

FLAGS = flags.FLAGS # 将解析后的命令行参数存储在 FLAGS 对象中。


def main(unused_argv): # 主函数，absl.app 会调用它。
  # unused_argv 未被使用。

  # Initialize environment and task.
  # 初始化环境和任务。
  env_cls = Environment # 将 Environment 类赋值给 env_cls。
  env = env_cls( # 创建 Environment 类的实例。
      FLAGS.assets_root, # 传递资源根目录路径。
      disp=FLAGS.disp, # 传递是否显示GUI的标志。
      shared_memory=FLAGS.shared_memory, # 传递是否使用共享内存的标志。
      hz=480, # 设置PyBullet物理仿真步长频率为480Hz。
      n_input_views=FLAGS.n_input_views) # 传递用于NeRF的输入视角数量。
  task = tasks.names[FLAGS.task](continuous=FLAGS.continuous) # 从 tasks.names 字典中根据命令行指定的任务名称获取任务类，并实例化。
                                                            # tasks.names 在 ravens/tasks/__init__.py 中定义。
  task.mode = FLAGS.mode # 设置任务的模式（如 'train' 或 'test'）。

  # Initialize scripted oracle agent and dataset.
  # 初始化脚本化的专家策略（oracle agent）和数据集对象。
  agent = task.oracle(env, steps_per_seg=FLAGS.steps_per_seg) # 获取指定任务的专家策略。oracle 方法在具体的任务类中定义。
  dataset = Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{task.mode}')) # 创建 Dataset 类的实例，用于存储数据。路径根据数据目录、任务名和模式生成。

  # Train seeds are even and test seeds are odd.
  # 训练集的随机种子是偶数，测试集的随机种子是奇数。
  seed = dataset.max_seed # 获取数据集中已存在的最大种子。
  if seed < 0: # 如果数据集是空的 (max_seed 初始化为 -1)。
    seed = -1 if (task.mode == 'test') else -2 # 测试模式种子从-1开始，训练模式从-2开始（之后会+2，所以实际从1或0开始）。
  
  # Collect training data from oracle demonstrations.
  # 从专家演示中收集训练数据。
  while dataset.n_episodes < FLAGS.n: # 循环直到收集到足够数量的回合 (FLAGS.n)。
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{FLAGS.n}') # 打印当前收集进度。
    episode, total_reward = [], 0 # 初始化当前回合的数据列表和总奖励。
    seed += 2 # 确保训练和测试种子根据模式交替（偶数/奇数）。
    np.random.seed(seed) # 设置 numpy 的随机种子，确保可复现性。
    env.set_task(task) # 在环境中设置当前任务。
    obs, info = env.reset() # 重置环境，获取初始观测 (obs) 和信息 (info)。
    reward = 0 # 初始化当前步的奖励。

    # Let agent act.
    # 让专家智能体执行动作。
    act = agent.act(obs, info) # 专家智能体根据当前观测和信息决定动作。
    episode.append((obs, act, reward, info)) # 将 (观测, 动作, 奖励, 信息) 元组添加到回合数据中。
                                            # 注意此时的 reward 是上一步的奖励，通常是0。
    obs, reward, done, info = env.step(act) # 在环境中执行动作，获取新的 (观测, 奖励, 是否结束, 信息)。
    episode.append((obs, None, reward, info)) # 将新的 (观测, None (因为这是回合的最后状态，没有下一个动作), 奖励, 信息) 添加到回合数据中。
    print(f'Total Reward: {reward} Done: {done}') # 打印该回合的总奖励和结束状态。对于这些通常一步完成的任务，reward 代表了成功与否。

    # Only save completed demonstrations.
    # 只保存成功完成的演示。
    if reward > 0.99 and not FLAGS.debug: # 如果奖励接近1 (表示成功) 并且不是调试模式。
      # Reset to the env before act.
      # 重置环境到执行动作之前的状态，以确保NeRF数据是在动作执行前的场景状态下收集的。
      np.random.seed(seed) # 重新设置随机种子以复现之前的场景。
      obs = env.reset() # 再次重置环境。

      # Save NeRF's training data.
      # 保存 NeRF 的训练数据。
      nerf_dataset_path = f'{dataset.path}/nerf-dataset/{dataset.n_episodes:06d}-{seed}' # 定义NeRF数据集的保存路径。
      if 'perspective' in FLAGS.task: # 如果任务名中包含 'perspective' (这似乎是一个特定的任务变种)。
        # Use a larger t (radius of camera arrays) to see the whole scene.
        # 使用更大的相机阵列半径 t，以便看到整个场景。
        write_nerf_data(nerf_dataset_path, env, act, t=0.8) # 调用 write_nerf_data 函数准备NeRF数据，t=0.8。
      else:
        write_nerf_data(nerf_dataset_path, env, act) # 对于其他任务，使用默认的 t 值调用 write_nerf_data。

      # Train NeRF to generate depth.
      # 训练 NeRF 以生成深度图（和彩色图）。
      n_steps = 5000 # NeRF 训练的迭代步数。
      os.makedirs(dataset.path, exist_ok=True) # 确保数据集的主路径存在。
      NGP_PATH = './orthographic-ngp' # 定义 orthographic-ngp (NeRF实现) 的路径。
      # 构建调用 NeRF 训练脚本的命令字符串。
      cmd = f'python {NGP_PATH}/scripts/run.py --mode nerf \
              --scene {nerf_dataset_path} \
              --width 160 --height 320 --n_steps {n_steps} \
              --screenshot_transforms {nerf_dataset_path}/test/transforms_test.json \
              --near_distance 0 \
              --nerfporter \
              --nerfporter_color_dir {dataset.path}/nerf-cmap/{dataset.n_episodes:06d}-{seed} \
              --nerfporter_depth_dir {dataset.path}/nerf-hmap/{dataset.n_episodes:06d}-{seed} \
              --screenshot_spp 1'
      # 参数解释：
      # --mode nerf: NeRF 模式。
      # --scene: NeRF 训练数据（由 write_nerf_data 生成的相机内外参和图像）的路径。
      # --width 160 --height 320: NeRF 训练和渲染的图像尺寸。
      # --n_steps: 训练步数。
      # --screenshot_transforms: 一个 JSON 文件路径，定义了需要从训练好的 NeRF 模型中渲染出来的（正交）视图的相机参数。
      # --near_distance 0: NeRF 渲染的近裁剪面。
      # --nerfporter: 一个特定于 MIRA/NeRF-Porter 的标志。
      # --nerfporter_color_dir 和 --nerfporter_depth_dir: 指定渲染出来的正交彩色图和高度图的保存目录。
      # --screenshot_spp 1: 每个像素的采样数 (Samples Per Pixel) 为1。
      os.system(cmd) # 执行该命令，这将调用外部的 orthographic-ngp 脚本来训练NeRF并渲染图像。

      # Store demonstrations.
      # 存储演示数据。
      dataset.add(seed, episode) # 将当前回合的数据 (episode) 和种子 (seed) 添加到 Dataset 对象中，Dataset 会将其保存到磁盘。

if __name__ == '__main__': # Python 脚本的入口点。
  app.run(main) # 运行 absl.app 定义的主函数。