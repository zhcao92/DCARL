Header header
#RA - RD - FA - FD eight radars.
uint8 channel
# one radar detect
float32 single_dist
### two radars are one and next, ie. ra - rb.
#two radars are one and next, ie. rb(send) - rc(receive).
float32 pair1_dist
### two radars are one and after next, ie. ra - rc.
#two radars are one and before, ie. rb(send) - ra(receive).
float32 pair2_dist

