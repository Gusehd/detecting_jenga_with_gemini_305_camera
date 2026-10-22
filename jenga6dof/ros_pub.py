"""ROS2 publisher for block 6DOF results (optional: no-op when rclpy is unavailable).

Topics (frame_id "jenga_tower", see config/tower.yaml for the tower frame):
  /jenga/block_pose          geometry_msgs/PoseStamped   estimated block pose (tower frame)
  /jenga/block_nominal_pose  geometry_msgs/PoseStamped   nominal block pose (tower frame)
  /jenga/block_delta         std_msgs/Float64MultiArray   w.r.t. the reference tower
  /jenga/block_delta_rel     std_msgs/Float64MultiArray   w.r.t. layers n-2..n+2
      [n, m, dx, dy, dz (mm), roll, pitch, yaw (deg), n_samples, valid]
      deltas are expressed in the block frame (x = long axis, y = width, z = up)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

TOWER_FRAME = "jenga_tower"


class BlockPublisher:
    def __init__(self, enabled=True):
        self.node = None
        if not enabled:
            return
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import Float64MultiArray
        except ImportError as e:
            print(f"[ros] rclpy unavailable ({e}); publishing disabled")
            return
        self._PoseStamped, self._Float64MultiArray = PoseStamped, Float64MultiArray
        rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node("jenga_block_6dof")
        self.pub_pose = self.node.create_publisher(PoseStamped, "/jenga/block_pose", 10)
        self.pub_nom = self.node.create_publisher(PoseStamped, "/jenga/block_nominal_pose", 10)
        self.pub_delta = self.node.create_publisher(Float64MultiArray, "/jenga/block_delta", 10)
        self.pub_rel = self.node.create_publisher(Float64MultiArray, "/jenga/block_delta_rel", 10)
        print("[ros] publishing /jenga/block_pose, /jenga/block_nominal_pose, "
              "/jenga/block_delta, /jenga/block_delta_rel")

    def _pose_msg(self, T):
        msg = self._PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = TOWER_FRAME
        p, q = T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
        return msg

    def _delta_msg(self, n, m, D, n_samples, valid):
        D = D if valid else np.eye(4)
        rpy = R.from_matrix(D[:3, :3]).as_euler("xyz", degrees=True)
        msg = self._Float64MultiArray()
        msg.data = [float(n), float(m), *map(float, D[:3, 3] * 1000), *map(float, rpy),
                    float(n_samples), float(valid)]
        return msg

    def publish(self, n, m, T_nominal, delta, n_samples, D_rel=None):
        """delta: 4x4 block-frame delta w.r.t. the reference (None if no valid estimate);
        D_rel: same, relative to the neighbouring layers."""
        if self.node is None:
            return
        valid = delta is not None
        self.pub_delta.publish(self._delta_msg(n, m, delta, n_samples, valid))
        self.pub_rel.publish(self._delta_msg(n, m, D_rel, n_samples, D_rel is not None))
        self.pub_nom.publish(self._pose_msg(T_nominal))
        if valid:
            self.pub_pose.publish(self._pose_msg(T_nominal @ delta))
        self._rclpy.spin_once(self.node, timeout_sec=0)

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
            self._rclpy.shutdown()
