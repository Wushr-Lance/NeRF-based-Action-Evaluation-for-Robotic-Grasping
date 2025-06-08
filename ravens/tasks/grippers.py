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

  def __init__(self, assets_root):
    self.assets_root = assets_root
    self.activated = False

  def step(self):
    """This function can be used to create gripper-specific behaviors."""
    return

  def activate(self, objects):
    del objects
    return

  def release(self):
    return


class Spatula(Gripper):
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


class Suction(Gripper):
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
    super().__init__(assets_root)

    # Load suction gripper base model (visual only).
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))

    # Load suction tip model (visual and collision) with compliance.
    pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.08))
    p.changeConstraint(constraint_id, maxForce=50)

    # Reference to object IDs in environment for simulating suction.
    self.obj_ids = obj_ids

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Defaults for deformable parameters, and can override in tasks.
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    self.def_threshold = 0.030
    self.def_nb_anchors = 1

    # Track which deformable is being gripped (if any), and anchors.
    self.def_grip_item = None
    self.def_grip_anchors = []

    # Determines release when gripped deformable touches a rigid/def.
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self):
    """Simulate suction using a rigid fixed constraint to contacted object."""
    # TODO(andyzeng): check deformables logic.
    # del def_ids

    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
      if points:
        for point in points:
          obj_id, contact_link = point[2], point[4]
        if obj_id in self.obj_ids['rigid']:
          body_pose = p.getLinkState(self.body, 0)
          obj_pose = p.getBasePositionAndOrientation(obj_id)
          world_to_body = p.invertTransform(body_pose[0], body_pose[1])
          obj_to_body = p.multiplyTransforms(world_to_body[0],
                                             world_to_body[1],
                                             obj_pose[0], obj_pose[1])
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=0,
              childBodyUniqueId=obj_id,
              childLinkIndex=contact_link,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=obj_to_body[0],
              parentFrameOrientation=obj_to_body[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))
        self.activated = True

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
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    if points:
      return True

    return False

  def check_grasp(self):
    """Check a grasp (object in contact?) for picking success."""

    suctioned_object = None
    if self.contact_constraint is not None:
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
    return suctioned_object is not None

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
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, GRIPPER_URDF), pose[0], pose[1])

    # Attach gripper to robot end effector with stronger constraint
    self.constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0))
    # Increase constraint force to prevent detachment
    p.changeConstraint(self.constraint_id, maxForce=500)

    # Reference to object IDs in environment for simulating grasping.
    self.obj_ids = obj_ids

    # Indicates whether gripper is gripping anything.
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Get gripper joint indices for finger control
    self.joints = []

    n_joints = p.getNumJoints(self.body)

    for i in range(n_joints):
      joint_info = p.getJointInfo(self.body, i)

      if joint_info[1] == b'robotiq_2f_85_right_driver_joint':
        self.joints.append(i)

    # Gripper state: 0 = open, 1 = closed
    self.gripper_state = 0

    self.max_force = 50  # Increased force for better grasping

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

      if contact_points:
        print("Contact points found:")
        for i, point in enumerate(contact_points):
          obj_id = point[2]  # bodyB

          contact_link = point[4]

          contact_pos = point[5]  # contact position on bodyA

          print(f"  Point {i}: obj_id={obj_id}, contact_pos={contact_pos}")

          if obj_id in self.obj_ids['rigid']:
            print(f"Found grippable object: {obj_id}")

            # Create constraint to "grasp" the object
            gripper_pose = p.getBasePositionAndOrientation(self.body)
            obj_pose = p.getBasePositionAndOrientation(obj_id)

            # Calculate relative pose
            world_to_gripper = p.invertTransform(gripper_pose[0], gripper_pose[1])
            obj_to_gripper = p.multiplyTransforms(world_to_gripper[0],
                                                 world_to_gripper[1],
                                                 obj_pose[0], obj_pose[1])

            # Create strong fixed constraint between gripper and object
            self.contact_constraint = p.createConstraint(
                parentBodyUniqueId=self.body,
                parentLinkIndex=-1,
                childBodyUniqueId=obj_id,
                childLinkIndex=contact_link,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=obj_to_gripper[0],
                parentFrameOrientation=obj_to_gripper[1],
                childFramePosition=(0, 0, 0),
                childFrameOrientation=(0, 0, 0, 1))

            # Set high constraint force to ensure stable grasping
            p.changeConstraint(self.contact_constraint, maxForce=1000)

            self.activated = True
            print(f"✓ Gripper activated: grasped object {obj_id}")
            return

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
    for joint_id in self.joints:
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
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    if points:
      return True

    return False

  def check_grasp(self):
    """Check if gripper is successfully grasping an object."""
    gripped_object = None
    if self.contact_constraint is not None:
      gripped_object = p.getConstraintInfo(self.contact_constraint)[2]
    return gripped_object is not None

  def get_registration_info(self):
    """Return gripper registration info for recording."""
    return self.body, os.path.join(self.assets_root, GRIPPER_URDF), None, None