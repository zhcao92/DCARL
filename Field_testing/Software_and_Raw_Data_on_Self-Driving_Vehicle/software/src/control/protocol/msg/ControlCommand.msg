# Universal vehicle dynamics command output. This message is sent in high frequency

Header header

# TODO(zyxin): should these params be percentage or physical values?
# Level of accelaration, unit in m/s^2, throttle is positive, braking is negative
float32 accel
# Level of steering on front wheel, unit in radian, left turning is positive
float32 steer

# Gear shift, positive means manual transmission, negative means automatic transmission
int8 gear
int8 GEAR_NONE = 0
int8 GEAR_NEUTRAL = -1
int8 GEAR_REVERSE = -2
int8 GEAR_PARKING = -3
int8 GEAR_DRIVE = -4

# Parking brake switch
bool parking_brake
