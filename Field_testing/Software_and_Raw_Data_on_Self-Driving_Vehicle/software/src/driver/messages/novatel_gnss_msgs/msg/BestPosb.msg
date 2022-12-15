Header header

#novatel long header
LongHeader long_header

uint8 solution_status
uint8 position_type
# in degrees
float64 latitude      
# in degrees                 
float64 longitude     
# height above mean sea level in meters              
float64 height_msl   
# undulation = height_wgs84 - height_msl             
float32 undulation               
# datum id number
uint8 datum_id              
# latitude standard deviation (m)
float32 latitude_std_dev        
# longitude standard deviation (m)
float32 longitude_std_dev       
# height standard deviation (m)
float32 height_std_dev          
# base station id      totally 4 char
string base_station_id       
# differential position age (sec)
float32 differential_age             
# solution age (sec)
float32 solution_age                     
# number of satellites tracked
uint8 num_sats_tracked            
# number of satellites used in solution
uint8 num_sats_in_solution   
# number of L1/E1/B1 satellites used in solution
uint8 num_sats_l1                  
# number of multi-frequency satellites used in solution
uint8 num_sats_multi         
# reserved
uint8 reserved                         
# extended solution status - OEMV and greater only
uint8 extended_solution_status       
uint8 galileo_beidou_used_mask
uint8 gps_glonass_used_mask
