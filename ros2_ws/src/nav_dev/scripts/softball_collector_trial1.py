#!/usr/bin/env python3

import math

import os

import time

 

import cv2

import rclpy

from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image

from nav_msgs.msg import Odometry

from geometry_msgs.msg import Twist

from cv_bridge import CvBridge

from ultralytics import YOLO

 

# ===== 調整パラメータ =====

IMAGE_W = 640          # カメラ画像の幅 [px]

CONF = 0.15            # YOLOの信頼度しきい値

SEARCH_W = 0.3         # 探索時の旋回速度 [rad/s]

APPROACH_V = 0.10      # 接近時の前進速度 [m/s]

TURN_GAIN = 0.004      # 画像中心からのずれ -> 旋回速度

CENTER_TOL = 80        # このずれ[px]以内なら前進する

CLOSE_WIDTH = 70       # 検出枠の幅がこれ以上で「捕捉」とみなす [px]

ESCAPE_WIDTH = 40      # DELIVER中にこれより小さく見えたら「ボールが離れた」

LOST_TIMEOUT = 1.5     # 検出がこの秒数途切れたら見失いとみなす

GOAL_X, GOAL_Y = 3.0, 0.0   # 回収エリア中心（world.sdfの位置）

PUSH_V = 0.08          # 押して運ぶときの前進速度 [m/s]

PUSH_W_MAX = 0.2       # 押して運ぶときの最大旋回速度 [rad/s]

GOAL_TOL = 0.45        # ロボットと回収エリア中心の距離がこれ以下で到着 [m]

 

 

class SoftballCollector(Node):

 

    def __init__(self):

        super().__init__('softball_collector')

        self.model = YOLO(os.path.expanduser('~/yolov8n.pt'))

        self.bridge = CvBridge()

        self.create_subscription(

            Image, '/camera', self.on_image, qos_profile_sensor_data)

        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.img_pub = self.create_publisher(

            Image, '/softball_detection/image', 10)

        self.create_timer(0.1, self.control)

        self.state = 'SEARCH'

        self.det = None          # (中心x, 幅, 検出時刻)

        self.x = None

        self.y = None

        self.yaw = None

        self.done_time = None

        self.get_logger().info('Softball collector started: SEARCH')

 

    def on_odom(self, msg):

        p = msg.pose.pose.position

        q = msg.pose.pose.orientation

        self.x, self.y = p.x, p.y

        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),

                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

 

    def on_image(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        result = self.model.predict(

            frame, classes=[32], conf=CONF, verbose=False)[0]

        out = result.plot()

        if len(result.boxes) > 0:

            box = max(result.boxes, key=lambda b: float(b.conf[0]))

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            self.det = ((x1 + x2) / 2.0, x2 - x1, time.time())

        cv2.putText(out, f'STATE: {self.state}', (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        img = self.bridge.cv2_to_imgmsg(out, 'bgr8')

        img.header = msg.header

        self.img_pub.publish(img)

 

    def set_state(self, new_state):

        if new_state != self.state:

            self.get_logger().info(f'{self.state} -> {new_state}')

            self.state = new_state

 

    def control(self):

        cmd = Twist()

        now = time.time()

        fresh = self.det is not None and now - self.det[2] < LOST_TIMEOUT

 

        if self.state == 'SEARCH':

            if fresh:

                self.set_state('APPROACH')

            else:

                cmd.angular.z = SEARCH_W

 

        elif self.state == 'APPROACH':

            if not fresh:

                self.set_state('SEARCH')

            else:

                cx, width, _ = self.det

                err = IMAGE_W / 2.0 - cx

                cmd.angular.z = max(-0.5, min(0.5, TURN_GAIN * err))

                if abs(err) < CENTER_TOL:

                    cmd.linear.x = APPROACH_V

                if width >= CLOSE_WIDTH:

                    self.set_state('DELIVER')

 
        elif self.state == 'DELIVER':

            if self.x is None:

                return

            # 回収エリアまでの距離

            dx = GOAL_X - self.x

            dy = GOAL_Y - self.y

            distance = math.hypot(dx, dy)

            # 回収エリア付近まで来たら終了

            if distance < GOAL_TOL:

                self.done_time = now

                self.set_state('DONE')

            # ボールを見失ったら、もう一度追跡

            elif not fresh:

                self.set_state('APPROACH')

            else:

                cx, width, _ = self.det

                # ボールが画面中央からどれだけずれているか

                ball_error = IMAGE_W / 2.0 - cx

                # ボールが常に正面に来るよう旋回

                cmd.angular.z = max(

                    -0.20,

                    min(0.20, TURN_GAIN * ball_error)

                )

                # ボールが正面付近にあるときだけ前進

                if abs(ball_error) < CENTER_TOL:

                    cmd.linear.x = PUSH_V



        elif self.state == 'DONE':

            if now - self.done_time < 1.5:

                cmd.linear.x = -0.08     # 少し下がってボールを置いていく

 

        self.cmd_pub.publish(cmd)

 

 

def main(args=None):

    rclpy.init(args=args)

    node = SoftballCollector()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.cmd_pub.publish(Twist())   # 終了時に停止

        node.destroy_node()

        rclpy.shutdown()

 

 

if __name__ == '__main__':

    main()

