How to contribute
=================

Please follow the requirement presented in :doc:`/principle`

Readme Contents
###############

- For message definition packages: Usually related comments are contained in `.msg` files and no readme is needed.
- For algorithm implementation packages: One needs `Module IO`(input topic, output topic, parameters), `Reference`, etc.

ROS dependency separation
#########################

In order for easier use, we will try to offer non-ROS mode. The existence of environment variable `ZZZ_ROS` determines whether the code is built upon ROS or not. To achieve this flexity, each code should have several principles:

* Majority of the functional code should be wrapped by a ROS-independent package
* A separate file provide ros wrapper for the code to work as a node
* The main implementation (includes python binding) is put into `src` and the wrapper codes should be included in sources under `nodes`
* Replacement package will be generated for logging and messsage definition. By this way, we can provide a fully independent environment for each package.

External materials for starting up
##################################

- `ROS CMakeLists.txt documentation <http://wiki.ros.org/catkin/CMakeLists.txt>`_
- `How to structure a Python-based ROS package <http://www.artificialhumancompanions.com/structure-python-based-ros-package/>`_
