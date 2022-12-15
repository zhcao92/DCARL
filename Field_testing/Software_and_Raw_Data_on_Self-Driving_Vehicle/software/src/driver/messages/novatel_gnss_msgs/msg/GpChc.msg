Header header

#GPCHC header
string chc_header
#GNSS week
uint8 gps_week
#GNSS seconds from week start
float32 gps_time
#heading degree
float32 heading
#pitch degree
float32 pitch
#roll degree
float32 roll
#gyroscope x axis
float32 gyro_x
#gyroscope y axis
float32 gyro_y
#gyroscope z axis
float32 gyro_z
#acceleration x axis
float32 acc_x
float32 acc_y
float32 acc_z
float32 latitude
float32 longitude
float32 altitude
float32 vel_east
float32 vel_north
float32 vel_up
#velocity of vehicle
float32 vel_vehicle
#satellite number in main antenna 1
uint8 NSV1
#satellite number in secondary antenna 2
uint8 NSV2
#system status 
#lower 4 bits : 0-inital ; 1-卫导模式; 2-组合导航模式; 3-纯惯导模式;
#hign 4 bits : 0-不定位不定向 ; 1-单点定位定向; 2-单点定位定向; 3-组合推算; 
#4-RTK 稳定解定位定向; 5-RTK浮点解定位定向; 6-单点定位不定向; 
#7-伪距差分定位不定向; 5-RTK 稳定解定位不定向; 9-RTK 浮点解定位不定向.
uint8 status
#differential delay
uint8 age
#bit0:1:no GPS message, 0: normal
#bit1:1:no vehicle message, 0: normal
#bit3:1:gyroscope error, 0: normal
#bit4:1:accelerometer error, 0: normal
uint8 warning
#check sum
uint8 checksum

