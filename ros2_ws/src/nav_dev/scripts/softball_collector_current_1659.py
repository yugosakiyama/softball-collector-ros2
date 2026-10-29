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

SEARCH_W = 0.30

APPROACH_V = 0.16

TURN_GAIN = 0.004

CENTER_TOL = 60

NEAR_DIST = 0.70       # カメラからボールまでこの距離で停止（触れない）

 

# ===== 回り込み =====

GOAL_X, GOAL_Y = 3.0, 0.0

STAGE_DIST = 0.45      # ボール中心の何m後方に回り込むか

LINE_TOL = 0.03

BEHIND_TOL = 0.05

STAGE_TOL = 0.10

MOVE_V = 0.28

ALIGN_TOL = 0.05       # [rad]

 

# ===== 押し運び =====

PUSH_V = 0.20

PUSH_HEAD_GAIN = 0.8

PUSH_BALL_GAIN = 0.0015

PUSH_W_MAX = 0.12

ESCAPE_DIST = 0.45     # 押している最中にボールがこれ以上離れたら回り込み直す

GOAL_TOL = 0.30

GOAL_HALF = 0.35     #ボールが緑の板の中心から±0.35m以内に入ったら完了

PUSH_LOST_DIST = 0.70       #ロボットとボールの距離がこれより開いたら回り込み直す[m]

REALIGN_ANGLE = 0.45        #押す向きのずれがこれ以上なら回り込み直す[rad](約25度)

LAT_GAIN = 1.5
 

 

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

        self.done_x = None

        self.done_y = None

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

                err = IMAGE_W / 2.0 - cx

                distance_to_ball = None

                if (

                    self.ball_x is not None

                    and self.ball_y is not None

                    and self.x is not None

                    and self.y is not None

                ):

                    distance_to_ball = math.hypot(

                        self.ball_x - self.x,

                        self.ball_y - self.y

                    )

                # First align with the ball.

                cmd.angular.z = clamp(

                    TURN_GAIN * err,

                    -0.4,

                    0.4

                )

                # Close enough: calculate the stage point and start going around.

                if (

                    distance_to_ball is not None

                    and distance_to_ball <= NEAR_DIST

                    and abs(err) < CENTER_TOL

                ):

                    cmd.linear.x = 0.0

                    cmd.angular.z = 0.0

                    if self.calculate_stage():

                        self.set_state('GO_STAGE')

                # Move forward only while the ball is near the image center.

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

            if (

                self.ball_x is None

                or self.ball_y is None

                or self.x is None

                or self.y is None

                or self.yaw is None

            ):

                return

            bx = self.ball_x

            by = self.ball_y

            # Unit vector from the ball toward the goal.

            gx = GOAL_X - bx

            gy = GOAL_Y - by

            norm = math.hypot(gx, gy)

            if norm < 0.001:

                return

            ux = gx / norm

            uy = gy / norm

            # Vector from the ball to the robot.

            rx = self.x - bx

            ry = self.y - by

            # Positive / negative shows which side of the line the robot is on.

            lateral = -uy * rx + ux * ry

            # Positive value means the robot is behind the ball.

            behind = -(ux * rx + uy * ry)

            line_ok = abs(lateral) < LINE_TOL

            behind_ok = abs(behind - STAGE_DIST) < BEHIND_TOL

            # Move to the exact point 0.45 m behind the ball.

            target_x = bx - STAGE_DIST * ux

            target_y = by - STAGE_DIST * uy

            dx = target_x - self.x

            dy = target_y - self.y

            dist = math.hypot(dx, dy)

            # Do not proceed until both geometric conditions are satisfied.

            if line_ok and behind_ok:

                self.get_logger().info(

                    f'stage ready: lateral={lateral:+.3f} '

                    f'behind={behind:.3f}'

                )

                self.set_state('ALIGN')

            else:

                target_heading = math.atan2(dy, dx)

                e = angle_diff(target_heading, self.yaw)

                # First turn toward the target point.

                if abs(e) > 0.30:

                    cmd.angular.z = clamp(

                        1.0 * e,

                        -0.35,

                        0.35

                    )

                # Then move toward it with gentle steering.

                else:

                    cmd.angular.z = clamp(

                        0.8 * e,

                        -0.15,

                        0.15

                    )

                    cmd.linear.x = clamp(

                        0.5 * dist,

                        0.015,

                        MOVE_V

                    )

 

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

                    # Re-check the robot position after turning.

                    gx = GOAL_X - self.ball_x

                    gy = GOAL_Y - self.ball_y

                    norm = math.hypot(gx, gy)

                    if norm < 0.001:

                        return

                    ux = gx / norm

                    uy = gy / norm

                    rx = self.x - self.ball_x

                    ry = self.y - self.ball_y

                    lateral = -uy * rx + ux * ry

                    behind = -(ux * rx + uy * ry)

                    line_ok = abs(lateral) < LINE_TOL

                    behind_ok = (

                        abs(behind - STAGE_DIST) < BEHIND_TOL

                    )

                    self.get_logger().info(

                        f'align check: lateral={lateral:+.3f} '

                        f'behind={behind:.3f}'

                    )

                    if line_ok and behind_ok:

                        self.set_state('PUSH')

                    else:

                        # Turning shifted the robot away from the line.

                        self.set_state('GO_STAGE')

                else:

                    # その場で旋回して向きを合わせる

                    cmd.angular.z = clamp(

                        1.0 * e,

                        -0.25,

                        0.25

                    )

 

        elif self.state == 'PUSH':

            if self.ball_x is None or self.yaw is None:

                return

            bx, by = self.ball_x, self.ball_y

            to_goal = math.hypot(GOAL_X - bx, GOAL_Y - by)

            # done: ball near the center of the goal area

            if to_goal < 0.20:

                self.get_logger().info(

                    f'ball in goal: ({bx:.2f}, {by:.2f})')

                self.done_time = now

                self.done_x = self.x

                self.done_y = self.y

                self.set_state('DONE')

            else:

                t = math.atan2(GOAL_Y - by, GOAL_X - bx)

                rx, ry = self.x - bx, self.y - by

                dist = math.hypot(rx, ry)

                lateral = -math.sin(t) * rx + math.cos(t) * ry

                to_ball = math.atan2(by - self.y, bx - self.x)

                misalign = abs(angle_diff(t, to_ball))

                if dist > PUSH_LOST_DIST or misalign > REALIGN_ANGLE:

                    self.get_logger().info(

                        f'realign: dist={dist:.2f} '

                        f'misalign={misalign:.2f}')

                    self.set_state('BACKOFF')

                else:

                    desired = t - clamp(LAT_GAIN * lateral, -0.3, 0.3)

                    err = angle_diff(desired, self.yaw)

                    cmd.angular.z = clamp(

                        1.0 * err, -PUSH_W_MAX, PUSH_W_MAX)

                    if abs(err) < 0.25:

                        # Slow down gently near the goal.

                        if to_goal > 0.60:

                            cmd.linear.x = PUSH_V

                        elif to_goal > 0.30:

                            cmd.linear.x = 0.14

                        else:

                            cmd.linear.x = 0.10

 

        elif self.state == 'BACKOFF':

 

            if self.ball_x is None:

                return

            d = math.hypot(self.ball_x - self.x, self.ball_y - self.y)

            if d < 0.45:

                # back straight away from the ball, no turning

                cmd.linear.x = -0.06

            elif self.calculate_stage():

                self.set_state('GO_STAGE')

 

        elif self.state == 'DONE':

            if (

                self.done_x is None

                or self.done_y is None

            ):

                return

            # Distance traveled since entering DONE.

            retreat_dist = math.hypot(

                self.x - self.done_x,

                self.y - self.done_y

            )

            # Back away from the collection area.

            if retreat_dist < 0.55:

                cmd.linear.x = -0.16

            else:

                cmd.linear.x = 0.0



 

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

