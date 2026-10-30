Autonomous Softball Collector
ROS 2 / Gazebo上で、YOLOv8nによる物体検出を利用して ソフトボールを認識し、自律的に回収エリアまで運搬するシステムです。

Overview
ロボットに搭載したカメラ画像に対してYOLOv8nによる リアルタイム物体検出推論を行い、ソフトボールを sports ball クラスとして検出します。

AIによる検出結果を利用してボールへ接近した後、 Gazebo上のロボット・ボール位置情報を用いて回り込み位置を計算し、 ボールを緑色の回収エリアまで自律的に運搬します。

AI Inference
AI推論には学習済みYOLOv8nを使用しています。

処理の流れ：

Gazebo Camera
→ ROS 2 /camera
→ YOLOv8n inference
→ sports ball detection
→ SEARCH / APPROACH

YOLOによる推論結果から画像上のボール位置を取得し、 ロボットの探索・接近制御に利用します。

Autonomous Behavior
ロボットは以下の状態遷移によって自律回収を行います。

SEARCH
→ APPROACH
→ GO_STAGE
→ ALIGN
→ PUSH
→ BACKOFF
→ DONE

SEARCH
カメラ画像からボールを探索します。

APPROACH
YOLOv8nの検出結果を利用してボールへ接近します。

GO_STAGE
ボールと回収エリアの位置関係から回り込み位置を計算し、 ボール後方へ移動します。

ALIGN
ボールと回収エリアを結ぶ方向へロボットの向きを合わせます。

PUSH
ボールを保持しながら回収エリアへ運搬します。

BACKOFF
押し込み位置がずれた場合に一度後退して再度位置合わせを行います。

DONE
ボールが回収エリアへ入ったことを確認し、 ロボットのみ回収エリア外へ退避します。

Original Implementations
提供されたROS 2 / Gazeboサンプルをベースとして、 主に以下を追加・変更しました。

Gazebo上へのソフトボールモデル追加
緑色の回収エリア追加
ロボットへのカメラセンサ追加
YOLOv8nによるソフトボール検出
AI推論結果を利用したSEARCH / APPROACH
Gazebo dynamic poseを利用したロボット・ボール位置取得
ボールと回収エリアを結ぶ直線に基づく回り込み位置計算
lateral / behindによる位置合わせ判定
ボールの挙動を安定させるcollision・摩擦設定
ボール保持用フロントガード
PUSH / BACKOFF / DONEを含む自律状態遷移
回収完了後の自動退避
Environment
Ubuntu 22.04
ROS 2 Humble
Gazebo Fortress
Python 3.10
Ultralytics YOLOv8n
OpenCV
Build
cd ~/softball-collector-ros2/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

Run
Terminal 1
Shell
cd ~/softball-collector-ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch nav_dev nav_teleop.launch.py

Terminal 2
Shell
cd ~/softball-collector-ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run nav_dev softball_collector.py
