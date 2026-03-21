import subprocess
import time
import signal
import sys

procs = []

def cleanup(sig=None, frame=None):
    print("\nShutting down all nodes...")
    for p in procs:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

commands = [
    # (command_list, needs_own_terminal)
    (['ros2', 'launch', 'turtlebot3_bringup', 'robot.launch.py'], False),
    (['ros2', 'run', 'camera_ros', 'camera_node',
      '--ros-args', '-p', 'width:=640', '-p', 'height:=480', '-p', 'format:=BGR888'], False),
    (['ros2', 'run', 'tb3_exercise', 'video_publisher'], False),
    (['ros2', 'run', 'tb3_exercise', 'motion_server'], False),
]

# Start background nodes
for cmd, _ in commands:
    p = subprocess.Popen(cmd)
    procs.append(p)
    time.sleep(1.0)  # small stagger so bringup is first

# Teleop needs an interactive terminal — open it in a new window
time.sleep(3.0)  # wait for robot bringup
teleop = subprocess.Popen(
    ['xterm', '-e', 'ros2', 'run', 'turtlebot3_teleop', 'teleop_keyboard']
)
procs.append(teleop)

print("All nodes started. Press Ctrl+C to stop everything.")
signal.pause()  # wait for Ctrl+C