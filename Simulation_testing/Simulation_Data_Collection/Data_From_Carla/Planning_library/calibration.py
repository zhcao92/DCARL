import glob
import sys

sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg')[0])

import carla
import random
import math
import numpy as np


VEHICLE_TYPE = 'vehicle.lincoln.mkz_2020'
ACC_DELTA_T = 2
DEC_DELTA_T = 0.5
VECOLICY_GRID = np.linspace(1, 20, num=3, dtype=float)
THROTTLE_BRAKE_GRID = np.linspace(0, 1, num=3, dtype=float)
DT = 0.02
WAIT_TICK = 20

class Calibration:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world("Town04")
        self.world.set_weather(carla.WeatherParameters(cloudiness=0, precipitation=30.0, sun_altitude_angle=70.0))
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = DT
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        print("Carla Connected....")

        self.ego_vehicle = None
        self.blueprint_library = self.world.get_blueprint_library()
        # [ActorBlueprint(id=vehicle.audi.a2,tags=[vehicle, a2, audi]), 
        #  ActorBlueprint(id=vehicle.audi.tt,tags=[tt, vehicle, audi]), 
        #  ActorBlueprint(id=vehicle.micro.microlino,tags=[microlino, vehicle, micro]), 
        #  ActorBlueprint(id=vehicle.dodge.charger_police,tags=[charger_police, vehicle, dodge]), 
        #  ActorBlueprint(id=vehicle.jeep.wrangler_rubicon,tags=[wrangler_rubicon, vehicle, jeep]), 
        #  ActorBlueprint(id=vehicle.mini.cooper_s,tags=[cooper_s, vehicle, mini]), 
        #  ActorBlueprint(id=vehicle.nissan.micra,tags=[vehicle, micra, nissan]), 
        #  ActorBlueprint(id=vehicle.mercedes.coupe,tags=[coupe, vehicle, mercedes]), 
        #  ActorBlueprint(id=vehicle.seat.leon,tags=[leon, vehicle, seat]), 
        #  ActorBlueprint(id=vehicle.toyota.prius,tags=[prius, vehicle, toyota]), 
        #  ActorBlueprint(id=vehicle.harley-davidson.low_rider,tags=[low_rider, vehicle, harley-davidson]), 
        #  ActorBlueprint(id=vehicle.carlamotors.carlacola,tags=[carlacola, vehicle, carlamotors]), 
        #  ActorBlueprint(id=vehicle.mercedes.coupe_2020,tags=[coupe_2020, vehicle, mercedes]), 
        #  ActorBlueprint(id=vehicle.volkswagen.t2_2021,tags=[t2_2021, vehicle, volkswagen]), 
        #  ActorBlueprint(id=vehicle.bh.crossbike,tags=[crossbike, vehicle, bh]), 
        #  ActorBlueprint(id=vehicle.mini.cooper_s_2021,tags=[cooper_s_2021, vehicle, mini]), 
        #  ActorBlueprint(id=vehicle.dodge.charger_police_2020,tags=[charger_police_2020, vehicle, dodge]), 
        #  ActorBlueprint(id=vehicle.bmw.grandtourer,tags=[grandtourer, vehicle, bmw]), 
        #  ActorBlueprint(id=vehicle.mercedes.sprinter,tags=[sprinter, vehicle, mercedes]), 
        #  ActorBlueprint(id=vehicle.vespa.zx125,tags=[zx125, vehicle, vespa]), 
        #  ActorBlueprint(id=vehicle.yamaha.yzf,tags=[yzf, vehicle, yamaha]), 
        #  ActorBlueprint(id=vehicle.ford.crown,tags=[crown, vehicle, ford]), 
        #  ActorBlueprint(id=vehicle.ford.ambulance,tags=[ambulance, vehicle, ford]), 
        #  ActorBlueprint(id=vehicle.nissan.patrol_2021,tags=[vehicle, patrol_2021, nissan]), 
        #  ActorBlueprint(id=vehicle.nissan.patrol,tags=[patrol, vehicle, nissan]), 
        #  ActorBlueprint(id=vehicle.kawasaki.ninja,tags=[ninja, vehicle, kawasaki]), 
        #  ActorBlueprint(id=vehicle.lincoln.mkz_2020,tags=[mkz_2020, vehicle, lincoln]), 
        #  ActorBlueprint(id=vehicle.ford.mustang,tags=[mustang, vehicle, ford]), 
        #  ActorBlueprint(id=vehicle.lincoln.mkz_2017,tags=[mkz_2017, vehicle, lincoln]), 
        #  ActorBlueprint(id=vehicle.tesla.cybertruck,tags=[cybertruck, vehicle, tesla]),
        #  ActorBlueprint(id=vehicle.chevrolet.impala,tags=[impala, vehicle, chevrolet]), 
        #  ActorBlueprint(id=vehicle.volkswagen.t2,tags=[t2, vehicle, volkswagen]), 
        #  ActorBlueprint(id=vehicle.citroen.c3,tags=[vehicle, c3, citroen]), 
        #  ActorBlueprint(id=vehicle.diamondback.century,tags=[century, vehicle, diamondback]), 
        #  ActorBlueprint(id=vehicle.tesla.model3,tags=[model3, vehicle, tesla]), 
        #  ActorBlueprint(id=vehicle.carlamotors.firetruck,tags=[firetruck, vehicle, carlamotors]), 
        #  ActorBlueprint(id=vehicle.audi.etron,tags=[vehicle, etron, audi]), 
        #  ActorBlueprint(id=vehicle.dodge.charger_2020,tags=[charger_2020, vehicle, dodge]), 
        #  ActorBlueprint(id=vehicle.gazelle.omafiets,tags=[omafiets, vehicle, gazelle])]
        self.ego_bp = random.choice(self.blueprint_library.filter(VEHICLE_TYPE))
    
    def get_velocity(self):
        velocity = self.ego_vehicle.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
    def get_acceleration(self):
        acceleration = self.ego_vehicle.get_acceleration()
        return math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)

    def run_test(self):
        acc_table = []
        dec_table = []
        for velocity in VECOLICY_GRID:
            for target in THROTTLE_BRAKE_GRID:
                acc = self.acc_test(target, velocity)
                acc_table.append([velocity, target, acc])
                dec = self.dec_test(target, velocity)
                dec_table.append([velocity, target, dec])
        np.savetxt("acc_table.txt", acc_table, fmt="%f,%f,%f", newline='\n')
        np.savetxt("dec_table.txt", dec_table, fmt="%f,%f,%f", newline='\n')

    def acc_test(self, target, test_velocity):
        print("Running acceleration test: vehicle velocity: {}, throttle value:{}.".format(test_velocity, target))
        result = None
        self.ego_vehicle = self.world.spawn_actor(self.ego_bp, carla.Transform(
            carla.Location(-9.6, -205.3, 0.5),
            carla.Rotation(0, 90, 0)))
        
        
        for i in range(WAIT_TICK):
            self.ego_vehicle.set_target_velocity(carla.Vector3D(0, test_velocity * 1.1 ,0))
            self.world.tick()
        while True:
            trigger_velocity = self.get_velocity()
            # print("Current velocity : {}".format(trigger_velocity))
            self.world.tick()
            if trigger_velocity < test_velocity:
                start_velocoty = self.get_velocity()
                spent_time = 0
                for i in range(int(ACC_DELTA_T/DT)):
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle = target, brake = 0))
                    self.world.tick()
                    velocity = self.get_velocity()
                    acceleration = self.get_acceleration()
                    # if velocity < 0.001:
                    #     break
                    print("Current Velocity:{}, ACC:{}".format(velocity, acceleration))
                    spent_time += DT
                end_velocoty = self.get_velocity()
                result = (end_velocoty - start_velocoty) / spent_time
                print("Acceleration is {}. \n".format((end_velocoty - start_velocoty) / spent_time))
                break
        self.clear()
        return result
        
    def dec_test(self, target, test_velocity):
        print("Running deceleration test: vehicle velocity: {}, brake velue:{}.".format(test_velocity, target))
        result = None
        self.ego_vehicle = self.world.spawn_actor(self.ego_bp, carla.Transform(
            carla.Location(-9.6, -205.3, 0.5),
            carla.Rotation(0, 90, 0)))
        # *1.1 for intristic dec
        
        
        for i in range(WAIT_TICK):
            self.ego_vehicle.set_target_velocity(carla.Vector3D(0, test_velocity * 1.1 ,0))
            self.world.tick()
            
        while True:
            trigger_velocity = self.get_velocity()
            # print("Current velocity : {}".format(trigger_velocity))
            self.world.tick()
            if trigger_velocity < test_velocity:
                start_velocoty = self.get_velocity()
                spent_time = 0
                for i in range(int(DEC_DELTA_T/DT)):
                    self.ego_vehicle.apply_control(carla.VehicleControl(throttle = 0, brake = target))
                    self.world.tick()
                    spent_time += DT
                    velocity = self.get_velocity()
                    acceleration = self.get_acceleration()
                    if velocity < 0.001:
                        break
                    print("Current Velocity:{}, ACC:{}".format(velocity, acceleration))
                    
                end_velocoty = self.get_velocity()
                result = (end_velocoty - start_velocoty) / spent_time
                print("Deceleration is {}. \n".format((end_velocoty - start_velocoty) / spent_time))
                break
        self.clear()
        return result
    
    def clear(self):
        self.ego_vehicle.destroy()
        self.ego_vehicle = None


if __name__=="__main__":
    calib = Calibration()
    calib.run_test()
    # calib.acc_test(1.0, 1.073)