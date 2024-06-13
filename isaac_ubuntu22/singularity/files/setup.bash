# setup environment
export LD_LIBRARY_PATH=${HOME}/ibrido_ws/install/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${HOME}/ibrido_ws/install:$CMAKE_PREFIX_PATH
export PATH=${HOME}/ibrido_ws/install/bin:$PATH
export GAZEBO_PLUGIN_PATH=${HOME}/ibrido_ws/install/lib:$GAZEBO_PLUGIN_PATH
export PYTHONPATH=${HOME}/ibrido_ws/install/lib/python2.7/dist-packages:${HOME}/ibrido_ws/install/lib/python3.10/site-packages:${HOME}/ibrido_ws/install/lib/python3/dist-packages:$PYTHONPATH
export ROS_PACKAGE_PATH=${HOME}/ibrido_ws/ros_src:${HOME}/ibrido_ws/install/share:${HOME}/ibrido_ws/install/lib:$ROS_PACKAGE_PATH
export AMENT_PREFIX_PATH=${HOME}/ibrido_ws/install:$AMENT_PREFIX_PATH
export PKG_CONFIG_PATH=${HOME}/ibrido_ws/install/lib/pkgconfig:$PKG_CONFIG_PATH
