# This message contains barebone information sent as heartbeat. This information should be published every loop

# Contains sequential information. frame_id is the module id.
Header header

# Brief signal of the module status
int8 signal
int8 SIGNAL_NORMAL = 0
int8 SIGNAL_ERROR = -1
