# Readme of Supplementary Data File

This is the code repository for article "Continuous Improvement of Self-Driving Cars using Dynamic Confidence-Aware Reinforcement Learning"

[![DOI](https://zenodo.org/badge/578512035.svg)](https://zenodo.org/badge/latestdoi/578512035)

## 1. Structure

The files are structured as follows, where the note in the brackets briefly explains the content:
```
.
+-- Simulation_testing [Codes and Data in the article for Experiment 1 and 2]
|   +-- Simulation1
|   |   +-- test_DCARL.py [Run the results in Simulation1]
|   +-- Simulation2
|   |   +-- test_DCARL.py [Run the results in Simulation2]
|   +-- Simulation_Data_Collection
|   |   +-- Data_Sampling
|   |   |   +-- data_sampling.py [Collect More Simulated Data in a Distribution]
|   |   +-- Data_From_Carla
|   |   |   +-- test_value_collect.py [Collect More Simulated Data from CARLA]
+-- Field_Testing [Codes and Data in the article for Experiment 3]
|   +-- Software_and_Raw_Data_on_Self-Driving_Vehicle
|   |   +-- software [Codes on self-driving vehicles. It's a copy of CLAP framework which refer to https://gitlab.com/umvdl/zzz/zzz/-/tree/dev/xiaopengG3]
|   |   +-- run_docker.sh [DEMO codes]
|   |   +-- demo.rviz [Visual interface for the codes]
|   +-- Simulation1
|   |   +-- DrawData.m [Run for the results of scenario 1 in field testing]
|   +-- Simulation2
|   |   +-- DrawData.m [Run for the results of scenario 2 in field testing]
|   +-- Simulation3
|   |   +-- DrawData.m [Run for the results of scenario 3 in field testing]
```


## 2. Light DEMO Codes for Simulation 1 and 2

This DEMO shows the results of simulation 1 and 2.
The function is to estimate the confidence value of different policies and decide the final trajectory.

### 2.1 Dependencies

- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [matplotlib](https://pypi.org/project/matplotlib/)

### 2.2 Installation

The package is written in Python 3. The usage of the [Anaconda](https://www.anaconda.com/distribution/#download) Python distribution is recommended.

After you have installed Python 3.6.8 or above, open a terminal in the Supplementary Data File and execute the following command to install the software (including the dependencies):

```bash
pip install -r ./requirements.txt
```

### 2.3 Usage

- Run the results for simulation 1:

    ```bash
    python .\Simulation_testing\Simulation_1\test_DCARL.py
    ```

- Run the results for simulation 2:

    ```bash
    python .\Simulation_testing\Simulation_2\test_DCARL.py
    ```

### 2.4 More Data for evaluation

You can use more data for evaluation. We provide two ways for data collection

#### 2.4.1 Sampling-based Data Collection

Sampling the data from the true value of the state-action pair by running:

```bash
python .\Simulation_testing\Simulation_Data_Collection\Data_Sampling\data_sampling.py
```

#### 2.4.2 CARLA-based Data Collection

Sampling the data from the [CARLA](https://github.com/carla-simulator/carla/releases) simulation requires several steps:

- Download [CARLA](https://github.com/carla-simulator/carla/releases)>=0.9.13

- Run CARLA in the downloaded CARLA Folder

    ```bash
    .\CarlaUE4.sh
    ```

- Then, open another terminal and run 

    ```bash
    python .\Simulation_testing\Simulation_Data_Collection\Data_From_Carla\data_value_collect.py
    ```

#### 2.4.3 Instruction for other data:

A single data point should be formulated into {state, action, cumulative rewards}
Then, setting the original policy's action as the action 0.

After that, when given a state, the DCARL agent can generate an action according to the collected data. 


## 3. Field Testing Data and Codes

We provide the source codes on the autonomous vehicle and the data for three scenarios.

### 3.1 Dependencies

- This demo requires [ubuntu 18.04](https://releases.ubuntu.com/18.04/) and x86 platform.
- [docker](https://www.docker.com/)>=20.10.16
- [ROS melodic](http://wiki.ros.org/melodic/Installation)
- [rviz](http://wiki.ros.org/rviz)

### 3.2 Installation

#### 3.2.1 Install ros

Official tutotial can be found here: https://wiki.ros.org/melodic/Installation/Ubuntu

- Setup your sources.list 
    ```bash
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    ```

- Set up your keys 

    ```bash
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    ```

- update apt source

    ```bash
    sudo apt update
    ```

- install ros-melodic-desktop-full

    ```bash
    sudo apt install ros-melodic-desktop-full
    ```

- activate ros environment

    ```bash
    source /opt/ros/melodic/setup.bash
    ```

- rviz is default installed.

- This install takes about 20 minutes.

#### 3.2.2 Install docker

- Docker is default on ubuntu. If docker is not available, following this official tutorial to install docker on Linux: https://docs.docker.com/engine/install/ubuntu/.
- This install takes about 10 minutes.

#### 3.2.3 Download docker image

- The running environment is provided by docker image. Download the image from open docker hub.

    ```bash
    docker pull thuvehicle/dcarl_docker:v1
    ```

#### 3.2.4 Download field data records

- Download the field testing rosbag data from https://drive.google.com/file/d/1tTAyAlWsg4ZoeJVPpALbaXdlqhnm8P6X/view?usp=sharing.
- Unzip the file and copy the rosbags to desired location.

    ```bash
    unzip DCARL_data.zip
    cd data
    cp -r rosbag/ PATH_TO_PROJECT/supplementary_data_file/Field_testing/Software_and_Raw_Data_on_Self-Driving_Vehicle/software
    ```

### 3.3 Usages

#### 3.3.1 Steps

1. Run the docker image, and the source code is automatically mapped

    ```bash
    cd PATH_TO_PROJECT/supplementary_data_file/Field_testing/Software_and_Raw_Data_on_Self-Driving_Vehicle
    bash run_docker.sh
    ```

2. Now the terminal is inside the docker, enter the source location

    ```bash
    cd /zzz
    ```
Now, the running requirements is satisfied. We will use tmux to avoid opening multiple terminal.

3. Type `tmux` in the terminal, and enter the tmux environment

    Type `ctrl+b`, and press `c` to open a new terminal in tmux, we need 4 termials in total. We can see the number of opened terminal at bottom. The `ctrl+b` is a triggering operation in tmux. Before starting any operation of tmux, `ctrl+b` is required,

4. Type `ctrl+b`, and press number to swtich between different terminals.

5. In first tmux terminal, type `roscore` to start ros.

6. In second tmux terminal, we need to start the RL agent server by 

    ```bash
    cd src/tools/DCARL/
    python3 DCARL_agent.py
    ```

7. In third tmux terminal, start the planner by 

    ```bash
    launch_planner.sh
    ```

8. In forth tmux terminal, play the rosbag. Not the number after script, it indicates the number of case selected. 1, 2, and 3 is available.

    ```bash
    launch_rosbag.sh 1
    ```

9. At last, open a new terminal outside of docker to run RVIZ. We don't run rviz in docker because it may cause subtle bugs on different machine.

    ```bash
    rviz -d demo.rviz
    ```

10. Need to notify that it's required to restart the code in step 8 and step 9 if you want to switch between different bags by running `launch_rosbag.sh 1(2 or 3)`

#### 3.3.2 Expected run time

The operation before running takes about 10 minutes. To see all the demo cases requires 5 minutes.

#### 3.3.3 Expected output

The planning outputs trajectories online observing data record from rosbag. We can observed safe and reliable trajectories in RVIZ under different scenarios.

### 3.4 Collected Data Display for three Example Cases 

To redraw the Fig. 5 in the main text, we provide the source data of these cases and the codes for figures.

These codes are written in [Matlab](https://www.mathworks.com/products/matlab.html)>=R2020b.

Run the following code in Matlab for case 1:

```bash
\Field_testing\Scenario1\DrawData.m
```

Run the following code in Matlab for case 2:

```bash
\Field_testing\Scenario2\DrawData.m
```

Run the following code in Matlab for case 3:

```bash
\Field_testing\Scenario3\DrawData.m
```


