Header header
# where are the radars, 0 = rear 1 = front
uint8 where

# judge: no detect 0, 
uint8 judge

# cordination sort, dection point, auxiliary point.
uint8 cord_sort

# trace times
uint8 trace

# group, 0 - 9
uint8 group

#cordinate - mapx, unit metre
float32 map_x

# objs hight: 1, low: 0
uint8 mapz_judge

# cordinate - mapy, unit metre
float32 map_y
