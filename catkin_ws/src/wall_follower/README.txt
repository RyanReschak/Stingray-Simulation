This package controls the triton lidar robot in the stingray simulation
Prereqs: 
Download Stingray-simulation and put this package in the catkin/src folder.
Also install numpy, you can do this by doing in the terminal:
pip install numpy

To see this in action do the following:
Make sure to do a catkin_make and source devel/setup.bash 
in two different terminals. In one do 1) followed by 2) in the other terminal:
1) roslaunch stingray_sim wall_following_v1.launch
2) roslaunch wall_follower wall_following.launch

You can try running wall_following a few times to see different results
