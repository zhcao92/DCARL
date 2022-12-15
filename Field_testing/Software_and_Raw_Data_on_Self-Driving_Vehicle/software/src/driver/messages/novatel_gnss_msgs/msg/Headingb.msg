Header header

#novatel long header
LongHeader long_header

uint8 solution_status
uint8 position_type
float32 length
float32 heading
float32 pitch
float32 reserved
float32 heading_std_dev
float32 pitch_std_dev
# station id : 4 char
string station_id            
# number of satellites tracked
uint8 num_sats_tracked      
# number of satellites used in solution
uint8 num_sats_in_solution  
uint8 num_sats_ele
uint8 num_sats_l2
uint8 solution_source
uint8 extended_solution_status
uint8 galileo_beidou_sig_mask
uint8 gps_glonass_sig_mask
