class LaneAction(object):
    
    def __init__(self,target_lane_index=None,target_speed=None):
        self.target_lane_index = target_lane_index
        self.target_speed = target_speed

class TrajectoryAction(object):

    def __init__(self, trajectory=None, desired_speed=None):
        self.trajectory = trajectory
        self.desired_speed = desired_speed

class ControlAction(object):

    def __init__(self,acc=0,steering=0):
        self.acc = acc
        self.steering = steering