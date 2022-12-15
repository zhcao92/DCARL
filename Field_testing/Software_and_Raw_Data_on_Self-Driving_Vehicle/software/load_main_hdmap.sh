REL_PARENT_PATH=`dirname $0`
WORKSPACE_HOST_DIR=$(cd $(dirname $(cd $REL_PARENT_PATH; pwd)); pwd)
WORKSPACE_NAME=$(basename $WORKSPACE_HOST_DIR)
export ZZZ_ROOT=$WORKSPACE_HOST_DIR
source ~/.bashrc
export RUN_PLATFORM=xiaopeng
source devel/setup.bash
roslaunch config/launch/full_stack/main_hdmap.launch
