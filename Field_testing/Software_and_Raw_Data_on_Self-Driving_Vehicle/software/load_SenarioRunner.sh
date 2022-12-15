export ROOT_SCENARIO_RUNNER=~/zzz/src/driver/simulators/carla/carla_challenge/scenario_runner
export RUN_PLATFORM=carla
#export CARLA_ROOT=~/simulition/carla-0.9.6/Dist/CARLA_Shipping_0.9.6/LinuxNoEditor
export CARLA_ROOT=${ROOT_SCENARIO_RUNNER}/shared_dir/carla-python/carla
#export CARLA_ROOT=/home/zhangxinhong/carla-0.9.6/Dist/CARLA_Shipping_0.9.6/LinuxNoEditor
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
#export CARLA_AUTOWARE_ROOT=~/shared_dir/carla-autoware/autoware_launch
#export CARLA_MAPS_PATH=~/shared_dir/carla-autoware/autoware_data/maps
#source $CARLA_AUTOWARE_ROOT/../catkin_ws/devel/setup.bash
export PYTHONPATH=$PYTHONPATH:${ROOT_SCENARIO_RUNNER}/shared_dir/carla-python/carla/dist/carla-0.9.9-py2.7-linux-x86_64.egg:${ROOT_SCENARIO_RUNNER}/shared_dir/carla-python/carla/::${ROOT_SCENARIO_RUNNER}
#source /home/autoware/Autoware/install/setup.bash

export TEAM_AGENT=${ROOT_SCENARIO_RUNNER}/srunner/challenge/autoagents/ros_agent_clap.py
export TEAM_CODE_ROOT=~/zzz/src/driver/simulators/carla/carla_adapter/scripts/use_srunner
export TEAM_CONFIG=""
export MINI_AUTO_ROOT=/home/autoware/zzz
#python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py --scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios_test.json --routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training_test.xml --debug=1 --agent=${TEAM_AGENT} --config=${TEAM_CONFIG}


#export PYTHONPATH=$PYTHONPATH:~/simulation/carla-python/carla/dist/carla-0.9.6-py2.7-linux-x86_64.egg:~/simulation/carla-python/carla/
source devel/setup.bash
#python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes_zxh.py --scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios_test.json --routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training_test.xml --debug=1 --agent=${TEAM_AGENT} --config=${TEAM_CONFIG}
#python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes_temp.py --scenarios=${ROOT_SCENARIO_RUNNER}/srunner/CaseGenerating/dist/NovAutoTesting/Scenario_temp.json --routes=${ROOT_SCENARIO_RUNNER}/srunner/CaseGenerating/dist/NovAutoTesting/Routes_temp.xml --debug=1 --agent=${TEAM_AGENT} --config=${TEAM_CONFIG}
python ${ROOT_SCENARIO_RUNNER}/srunner/challenge/challenge_evaluator_routes.py --scenarios=${ROOT_SCENARIO_RUNNER}/srunner/challenge/all_towns_traffic_scenarios.json --routes=${ROOT_SCENARIO_RUNNER}/srunner/challenge/routes_training.xml --debug=1 --agent=${TEAM_AGENT} --config=${TEAM_CONFIG}

