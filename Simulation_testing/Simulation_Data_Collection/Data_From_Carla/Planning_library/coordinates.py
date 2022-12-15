import numpy as np
import math


class Coordinates:
    
    def __init__(self, x0 = 0, y0 = 0, yaw0 = 0):
        
        self.x0 = x0
        self.y0 = y0
        self.yaw0 = yaw0
        
    def transfer_coordinate(self,x,y,vx,vy,yaw):
        '''
        input: list
        output:
        '''
        
        
        rotation = np.array([[math.cos(-self.yaw0), -math.sin(-self.yaw0)],
                             [math.sin(-self.yaw0), math.cos(-self.yaw0)]])
        
        loc = rotation.dot(np.array([x-self.x0, y-self.y0]))
        
        x_t = loc[0]
        y_t = loc[1]
        
        v = rotation.dot(np.array([vx, vy]))
        vx_t = v[0]
        vy_t = v[1]
        
        yaw_t = yaw-self.yaw0
        return x_t, y_t, vx_t, vy_t, yaw_t
        
        
if __name__ == "__main__":
    
    ego_coordinate = Coordinates(5, 10, 0.25*math.pi)
    print(ego_coordinate.transfer_coordinate(10,10,-1,1,0.75*math.pi))