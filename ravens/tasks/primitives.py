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

"""Motion primitives."""

import numpy as np
from ravens.utils import utils


class PickPlace():
  """Pick and place primitive."""

  def __init__(self, height=0.32, speed=0.01):
    self.height, self.speed = height, speed

  def __call__(self, movej, movep, ee, pose0, pose1):
    """Execute pick and place primitive.

    Args:
      movej: function to move robot joints.
      movep: function to move robot end effector pose.
      ee: robot end effector.
      pose0: SE(3) picking pose.
      pose1: SE(3) placing pose.

    Returns:
      timeout: robot movement timed out if True.
    """

    pick_pose, place_pose = pose0, pose1

    # Execute picking primitive.
    prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
    postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
    prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
    postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
    timeout = movep(prepick_pose)

    # Move towards pick pose until contact is detected.
    delta = (np.float32([0, 0, -0.001]),
             utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
    targ_pose = prepick_pose
    while not ee.detect_contact():  # and target_pose[2] > 0:
      targ_pose = utils.multiply(targ_pose, delta)
      timeout |= movep(targ_pose)
      if timeout:
        return True

    # Activate end effector, move up, and check picking success.
    ee.activate()
    timeout |= movep(postpick_pose, self.speed)
    pick_success = ee.check_grasp()

    # Execute placing primitive if pick is successful.
    if pick_success:
      preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
      postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
      preplace_pose = utils.multiply(place_pose, preplace_to_place)
      postplace_pose = utils.multiply(place_pose, postplace_to_place)
      targ_pose = preplace_pose
      while not ee.detect_contact():
        targ_pose = utils.multiply(targ_pose, delta)
        timeout |= movep(targ_pose, self.speed)
        if timeout:
          return True
      ee.release()
      timeout |= movep(postplace_pose)

    # Move to prepick pose if pick is not successful.
    else:
      ee.release()
      timeout |= movep(prepick_pose)

    return timeout


class PickPlaceGripper(PickPlace):
  """Pick and place primitive for grippers."""

  def __init__(self, height=0.32, speed=0.01, grasp_height_offset=0.005):
    super().__init__(height, speed)
    self.grasp_height_offset = grasp_height_offset # Offset above object center for grasping

  def __call__(self, movej, movep, ee, pose0, pose1):
    """Execute pick and place primitive for a gripper.

    Args:
      movej: function to move robot joints.
      movep: function to move robot end effector pose.
      ee: robot end effector (should be a gripper like RobotiqGripper).
      pose0: SE(3) picking pose (center of the object, gripper oriented for grasp).
      pose1: SE(3) placing pose.

    Returns:
      timeout: robot movement timed out if True.
    """
    pick_target_pose, place_target_pose = pose0, pose1

    # --- Picking --- 
    # 1. Move to a pre-grasp position above the object.
    #    pose0 is assumed to be at the grasp height, oriented correctly.
    pre_grasp_offset = ((0, 0, self.height), (0, 0, 0, 1)) # Approach from well above
    pre_grasp_approach_pose = utils.multiply(pick_target_pose, pre_grasp_offset)
    timeout = movep(pre_grasp_approach_pose)
    if timeout: return True

    # 2. Open gripper (if not already open) - RobotiqGripper.release() handles this.
    #    The ee.release() is typically called if a previous grasp failed or at the start.
    #    Assuming gripper is open before starting a pick.
    #    If explicit open is needed: ee.release() or a specific open command.

    # 3. Move down to the grasp pose (pose0).
    #    pose0 should be calculated such that the gripper fingers are at the correct height
    #    to grasp the object, not necessarily the center of the gripper TCP.
    #    We might adjust pose0 slightly upwards to avoid collision before closing.
    actual_grasp_approach_pose = (pick_target_pose[0], pick_target_pose[1]) 
    timeout |= movep(actual_grasp_approach_pose)
    if timeout: 
      # ee.release() # Ensure gripper is open if timeout occurs before grasp
      return True

    # 4. Close gripper.
    ee.activate() # This should call RobotiqGripper.activate() -> _close_gripper()
                  # and handle constraint creation.
    timeout |= movep(actual_grasp_approach_pose, self.speed) # Short pause or slight movement after closing
    if timeout: 
      # ee.release()
      return True

    # 5. Move up to a post-grasp position.
    post_grasp_offset = ((0, 0, self.height), (0, 0, 0, 1))
    post_grasp_retreat_pose = utils.multiply(pick_target_pose, post_grasp_offset)
    timeout |= movep(post_grasp_retreat_pose, self.speed)
    if timeout: 
      # ee.release()
      return True

    pick_success = ee.check_grasp() # RobotiqGripper.check_grasp() checks the constraint.

    # --- Placing (if pick was successful) ---
    if pick_success:
      # 1. Move to a pre-place position above the target placement.
      pre_place_offset = ((0, 0, self.height), (0, 0, 0, 1))
      pre_place_approach_pose = utils.multiply(place_target_pose, pre_place_offset)
      timeout |= movep(pre_place_approach_pose, self.speed)
      if timeout: return True # Object is still grasped

      # 2. Move down to the place pose.
      timeout |= movep(place_target_pose, self.speed)
      if timeout: return True # Object is still grasped

      # 3. Open gripper.
      ee.release() # RobotiqGripper.release() -> _open_gripper() and remove constraint.
      
      # 4. Move to a post-place position (retreat upwards).
      post_place_offset = ((0, 0, self.height * 0.5), (0, 0, 0, 1)) # Retreat less far after place
      post_place_retreat_pose = utils.multiply(place_target_pose, post_place_offset)
      timeout |= movep(post_place_retreat_pose, self.speed)
      if timeout: return True

    else: # Pick failed
      ee.release() # Ensure gripper is open
      # Optionally, move back to an initial safe pose if pick failed
      timeout |= movep(pre_grasp_approach_pose, self.speed)
      return timeout # Return timeout status from the recovery move

    return timeout


def push(movej, movep, ee, pose0, pose1):  # pylint: disable=unused-argument
  """Execute pushing primitive.

  Args:
    movej: function to move robot joints.
    movep: function to move robot end effector pose.
    ee: robot end effector.
    pose0: SE(3) starting pose.
    pose1: SE(3) ending pose.

  Returns:
    timeout: robot movement timed out if True.
  """

  # Adjust push start and end positions.
  pos0 = np.float32((pose0[0][0], pose0[0][1], 0.005))
  pos1 = np.float32((pose1[0][0], pose1[0][1], 0.005))
  vec = np.float32(pos1) - np.float32(pos0)
  length = np.linalg.norm(vec)
  vec = vec / length
  pos0 -= vec * 0.02
  pos1 -= vec * 0.05

  # Align spatula against push direction.
  theta = np.arctan2(vec[1], vec[0])
  rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

  over0 = (pos0[0], pos0[1], 0.31)
  over1 = (pos1[0], pos1[1], 0.31)

  # Execute push.
  timeout = movep((over0, rot))
  timeout |= movep((pos0, rot))
  n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
  for _ in range(n_push):
    target = pos0 + vec * n_push * 0.01
    timeout |= movep((target, rot), speed=0.003)
  timeout |= movep((pos1, rot), speed=0.003)
  timeout |= movep((over1, rot))
  return timeout


class PickPlaceContinuous:
  """A continuous pick-and-place primitive."""

  def __init__(self, speed=0.01):
    self.speed = speed
    self.reset()

  def reset(self):
    self.s_bit = 0  # Tracks the suction state.

  def __call__(self, movej, movep, ee, action):
    del movej
    timeout = movep(action['move_cmd'], speed=self.speed)
    if timeout:
      return True
    if action['suction_cmd']:
      if self.s_bit:
        ee.release()
      else:
        ee.activate()
      self.s_bit = not self.s_bit
    return timeout


class PushContinuous:
  """A continuous pushing primitive."""

  def __init__(self, fast_speed=0.01, slow_speed=0.003):
    self.fast_speed = fast_speed
    self.slow_speed = slow_speed

  def reset(self):
    pass

  def __call__(self, movej, movep, ee, action):
    del movej, ee
    speed = self.slow_speed if action['slowdown_cmd'] else self.fast_speed
    timeout = movep(action['move_cmd'], speed=speed)
    return timeout
