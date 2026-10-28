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

from tf2_msgs.msg import TFMessage

 

# ===== カメラ・認識 =====

IMAGE_W = 640

FOCAL_PX = 320.0       # 焦点距離[px] = (640/2) / tan(1.57/2)

BALL_K = 77.3          # 距離[m] = BALL_K / 検出枠の幅[px]（較正で調整）

CAM_OFFSET = 0.15      # base_linkからカメラまでの前方距離 [m]

CONF = 0.15

LOST_TIMEOUT = 1.5

BALL_REF = (1.5, 0.4)  # 較正ログ用のボール実位置（制御には使わない）

 

# ===== 探索・接近 =====

SEARCH_W = 0.3

APPROACH_V = 0.06

TURN_GAIN = 0.004

CENTER_TOL = 60

NEAR_DIST = 0.70       # カメラからボールまでこの距離で停止（触れない）

 

# ===== 回り込み =====

GOAL_X, GOAL_Y = 3.0, 0.0

STAGE_DIST = 0.45      # ボール中心の何m後方に回り込むか

STAGE_TOL = 0.06

MOVE_V = 0.10

ALIGN_TOL = 0.05       # [rad]

 

# ===== 押し運び =====

PUSH_V = 0.05

PUSH_HEAD_GAIN = 0.8

PUSH_BALL_GAIN = 0.0015

PUSH_W_MAX = 0.25

ESCAPE_DIST = 0.45     # 押している最中にボールがこれ以上離れたら回り込み直す

GOAL_TOL = 0.30

 

 

def angle_diff(a, b):

    d = a - b

    return math.atan2(math.sin(d), math.cos(d))

 

 

def clamp(v, lo, hi):

    return max(lo, min(hi, v))

 

 

class SoftballCollector(Node):

 

    def __init__(self):

        super().__init__('softball_collector')

        self.model = YOLO(os.path.expanduser('~/yolov8n.pt'))

        self.bridge = CvBridge()

        self.create_subscription(

            Image, '/camera', self.on_image, qos_profile_sensor_data)

        self.create_subscription(

            TFMessage,

            '/dynamic_pose',

            self.on_dynamic_pose,

            10

        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.img_pub = self.create_publisher(

            Image, '/softball_detection/image', 10)

        self.create_timer(0.1, self.control)

        self.state = 'SEARCH'

        self.det = None

        self.x = self.y = self.yaw = None

        self.ball_x = None

        self.ball_y = None

        self.ball = None

        self.stage = None

        self.done_time = None

        self.get_logger().info('Softball collector started: SEARCH')

 
    def on_dynamic_pose(self, msg):

        for transform in msg.transforms:

            name = transform.child_frame_id

            translation = transform.transform.translation

            rotation = transform.transform.rotation

            # ロボット本体の実際のWorld座標

            if name == 'demoRobot':

                self.x = translation.x

                self.y = translation.y

                self.yaw = math.atan2(

                    2.0 * (

                        rotation.w * rotation.z

                        + rotation.x * rotation.y

                    ),

                    1.0 - 2.0 * (

                        rotation.y * rotation.y

                        + rotation.z * rotation.z

                    )

                )

            # ソフトボールの実際のWorld座標

            elif name == 'softball':

                self.ball_x = translation.x

                self.ball_y = translation.y

 

    def on_image(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        result = self.model.predict(

            frame, classes=[32], conf=CONF, verbose=False)[0]

        out = result.plot()

        if len(result.boxes) > 0:

            box = max(result.boxes, key=lambda b: float(b.conf[0]))

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            self.det = ((x1 + x2) / 2.0, max(x2 - x1, 1.0), time.time())

        cv2.putText(out, f'STATE: {self.state}', (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        img = self.bridge.cv2_to_imgmsg(out, 'bgr8')

        img.header = msg.header

        self.img_pub.publish(img)

 

    def set_state(self, new_state):

        if new_state != self.state:

            self.get_logger().info(f'{self.state} -> {new_state}')

            self.state = new_state

 

    def calculate_stage(self):

        if (

            self.ball_x is None

            or self.ball_y is None

            or self.x is None

            or self.y is None

        ):

            return False

        bx = self.ball_x

        by = self.ball_y

        # ボールから緑の回収エリアへ向かうベクトル

        gx = GOAL_X - bx

        gy = GOAL_Y - by

        norm = math.hypot(gx, gy)

        if norm < 0.001:

            return False

        # ボール → ゴール方向の単位ベクトル

        ux = gx / norm

        uy = gy / norm

        # ボールから見てゴールと反対側に

        # ロボットが回り込む地点を作る

        stage_x = bx - STAGE_DIST * ux

        stage_y = by - STAGE_DIST * uy

        self.ball = (bx, by)

        self.stage = (stage_x, stage_y)

        self.get_logger().info(

            f'robot=({self.x:.2f}, {self.y:.2f}) | '

            f'ball=({bx:.2f}, {by:.2f}) | '

            f'stage=({stage_x:.2f}, {stage_y:.2f}) | '

            f'goal=({GOAL_X:.2f}, {GOAL_Y:.2f})'

        )

        return True

 

    def control(self):

        cmd = Twist()

        now = time.time()

        fresh = self.det is not None and now - self.det[2] < LOST_TIMEOUT

        if self.x is None:

            self.cmd_pub.publish(cmd)

            return

 

        if self.state == 'SEARCH':

            if fresh:

                self.set_state('APPROACH')

            else:

                cmd.angular.z = SEARCH_W

 

        elif self.state == 'APPROACH':

            if not fresh:

                self.set_state('SEARCH')

            else:

                cx, w, _ = self.det

                # 画像上での左右ずれ

                err = IMAGE_W / 2.0 - cx

                # dynamic_poseを使った実距離

                distance_to_ball = None

                if (

                    self.ball_x is not None

                    and self.ball_y is not None

                ):

                    distance_to_ball = math.hypot(

                        self.ball_x - self.x,

                        self.ball_y - self.y

                    )

                # まずボールを画面中央へ

                cmd.angular.z = clamp(

                    TURN_GAIN * err,

                    -0.4,

                    0.4

                )

                # 十分近づいたら接近をやめて、

                # ボール後方のstage位置を計算

                if (

                    distance_to_ball is not None

                    and distance_to_ball <= NEAR_DIST

                    and abs(err) < CENTER_TOL

                ):

                    cmd.linear.x = 0.0

                    cmd.angular.z = 0.0

                    if self.calculate_stage():

                        self.set_state('GO_STAGE')

                # まだ遠ければ、中央に捉えながらゆっくり接近

                elif (

                    distance_to_ball is not None

                    and abs(err) < CENTER_TOL

                ):

                    cmd.linear.x = clamp(

                        APPROACH_V

                        * (

                            distance_to_ball

                            - NEAR_DIST

                            + 0.2

                        )

                        / 0.5,

                        0.02,

                        APPROACH_V

                    )

 

        elif self.state == 'GO_STAGE':

            sx, sy = self.stage

            dx, dy = sx - self.x, sy - self.y

            dist = math.hypot(dx, dy)

            if dist < STAGE_TOL:

                self.set_state('ALIGN')

            else:

                e = angle_diff(math.atan2(dy, dx), self.yaw)

                cmd.angular.z = clamp(1.5 * e, -0.5, 0.5)

                if abs(e) < 0.3:

                    cmd.linear.x = min(MOVE_V, 0.8 * dist + 0.02)

 

        elif self.state == 'ALIGN':

            if (

                self.ball_x is None

                or self.ball_y is None

                or self.yaw is None

            ):

                self.set_state('SEARCH')

            else:

                # dynamic_poseから取得した最新のボール位置を使用

                # ボール → 緑の回収エリアの方向を計算

                target = math.atan2(

                    GOAL_Y - self.ball_y,

                    GOAL_X - self.ball_x

                )

                # 現在のロボット向きとの差

                e = angle_diff(

                    target,

                    self.yaw

                )

                # 十分ゴール方向を向いたらPUSHへ

                if abs(e) < ALIGN_TOL:

                    self.set_state('PUSH')

                else:

                    # その場で旋回して向きを合わせる

                    cmd.angular.z = clamp(

                        1.2 * e,

                        -0.4,

                        0.4

                    )

 

        elif self.state == 'PUSH':

            gx, gy = GOAL_X - self.x, GOAL_Y - self.y

            if math.hypot(gx, gy) < GOAL_TOL:

                self.done_time = now

                self.set_state('DONE')

            elif not fresh:

                self.set_state('SEARCH')

            else:

                cx, w, _ = self.det

                if BALL_K / w > ESCAPE_DIST:

                    self.set_state('APPROACH')   # 離れたので回り込み直す

                else:

                    head = angle_diff(math.atan2(gy, gx), self.yaw)

                    berr = IMAGE_W / 2.0 - cx

                    cmd.angular.z = clamp(

                        PUSH_HEAD_GAIN * head + PUSH_BALL_GAIN * berr,

                        -PUSH_W_MAX, PUSH_W_MAX)

                    cmd.linear.x = PUSH_V

 

        elif self.state == 'DONE':

            if now - self.done_time < 1.5:

                cmd.linear.x = -0.08

 

        self.cmd_pub.publish(cmd)

 

 

def main(args=None):

    rclpy.init(args=args)

    node = SoftballCollector()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.cmd_pub.publish(Twist())

        node.destroy_node()

        rclpy.shutdown()

 

 

if __name__ == '__main__':

    main()

