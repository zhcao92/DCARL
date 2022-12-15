# This message provide interface for additional vehicle modules, e.g. ACC, BSD


# Forward Collison Warning status
uint8 fcw_flag
uint8 FCW_AVAILABLE = 1
uint8 FCW_ALERT = 2

# Automatic Emergency Brake status
uint8 aeb_flag
uint8 AEB_AVAILABLE = 1
uint8 AEB_PRECHARGE = 2
uint8 AEB_BRAKING = 4

# Adaptive Cruise Control status
uint8 acc_flag
uint8 ACC_AVAILABLE = 1
uint8 ACC_BRAKING = 2

# Lane Departure Warning status
uint8 ldw_flag
uint8 LDW_AVAILABLE = 1
uint8 LDW_LEFT_ALERT = 2
uint8 LDW_RIGHT_ALERT = 4

# Blind Spot Detection status
uint8 bsd_flag
uint8 BSD_AVAILABLE = 1
uint8 BSD_LEFT_ALERT = 2
uint8 BSD_RIGHT_ALERT = 4
