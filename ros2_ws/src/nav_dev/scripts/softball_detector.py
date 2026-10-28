#!/usr/bin/env python3

 

import os

 

import rclpy

from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data

 

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from ultralytics import YOLO

 

 

class SoftballDetector(Node):

 

    def __init__(self):

        super().__init__('softball_detector')

 

        model_path = os.path.expanduser('~/yolov8n.pt')

 

        self.get_logger().info(

            f'Loading YOLO model: {model_path}'

        )

 

        self.model = YOLO(model_path)

        self.bridge = CvBridge()

 

        self.subscription = self.create_subscription(

            Image,

            '/camera',

            self.image_callback,

            qos_profile_sensor_data

        )

 

        self.result_publisher = self.create_publisher(

            Image,

            '/softball_detection/image',

            10

        )

 

        self.no_detection_count = 0

 

        self.get_logger().info('Softball detector started')

        self.get_logger().info('Input  : /camera')

        self.get_logger().info(

            'Output : /softball_detection/image'

        )

 

    def image_callback(self, msg):

 

        try:

            frame = self.bridge.imgmsg_to_cv2(

                msg,

                desired_encoding='bgr8'

            )

 

        except Exception as error:

            self.get_logger().error(

                f'Image conversion failed: {error}'

            )

            return

 

        # COCO class 32 = sports ball

        results = self.model.predict(

            source=frame,

            classes=[32],

            conf=0.15,

            verbose=False

        )

 

        result = results[0]

 

        detected = len(result.boxes) > 0

 

        # YOLOが描画した検出枠付き画像

        annotated_frame = result.plot()

 

        if detected:

 

            self.no_detection_count = 0

 

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

 

                confidence = float(

                    box.conf[0].cpu()

                )

 

                center_x = int(

                    (x1 + x2) / 2

                )

 

                center_y = int(

                    (y1 + y2) / 2

                )

 

                width = int(

                    x2 - x1

                )

 

                height = int(

                    y2 - y1

                )

 

                self.get_logger().info(

                    'Sports ball detected | '

                    f'center=({center_x}, {center_y}) | '

                    f'size=({width}, {height}) | '

                    f'confidence={confidence:.2f}'

                )

 

        else:

 

            self.no_detection_count += 1

 

            # 毎フレーム表示するとログが大量に出るため間引く

            if self.no_detection_count % 30 == 0:

                self.get_logger().info(

                    'Sports ball not detected'

                )

 

        try:

 

            output_msg = self.bridge.cv2_to_imgmsg(

                annotated_frame,

                encoding='bgr8'

            )

 

            output_msg.header = msg.header

 

            self.result_publisher.publish(

                output_msg

            )

 

        except Exception as error:

 

            self.get_logger().error(

                f'Result publishing failed: {error}'

            )

 

 

def main(args=None):

 

    rclpy.init(args=args)

 

    node = SoftballDetector()

 

    try:

        rclpy.spin(node)

 

    except KeyboardInterrupt:

        pass

 

    finally:

        node.destroy_node()

        rclpy.shutdown()

 

 

if __name__ == '__main__':

    main()
