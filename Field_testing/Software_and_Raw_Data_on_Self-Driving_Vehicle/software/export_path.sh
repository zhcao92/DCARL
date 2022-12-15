# export UE4_ROOT=/home/yli/UnrealEngine
export CARLA_ROOT=~/carla-0.9.9/Dist/CARLA_Shipping_0.9.9-69-gd26f6995-dirty/LinuxNoEditor
#export CARLA_ROOT=/home/yli/carla-0.9.5/Dist/CARLA_Shipping_0.9.5/LinuxNoEditor
export CARLA_VERSION=0.9.9
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py2.7-linux-x86_64.egg:$PYTHONPATH
export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/dist-packages:$PYTHONPATH
export PYTHONPATH=/usr/share/sumo/tools:${CARLA_ROOT}/PythonAPI/carla:$PYTHONPATH
export SUMO_HOME=/usr/share/sumo
