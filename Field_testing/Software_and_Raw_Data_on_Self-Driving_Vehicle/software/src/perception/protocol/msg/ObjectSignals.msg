# This message is used to represent detected vehicle light signals or human hand signals

# Signal flags. Multiple signal emission can exists in the same time.
uint16 flags

uint16 UNKNOWN                          = 0     # 0x00
uint16 NONE                             = 16    # 0x10

# This field is related to https://en.wikipedia.org/wiki/Automotive_lighting
uint16 VEHICLE_SIGNAL                   = 1     # 0x01
uint16 VEHICLE_SIGNAL_LEFT_TURN         = 17    # 0x11
uint16 VEHICLE_SIGNAL_RIGHT_TURN        = 33    # 0x21
uint16 VEHICLE_SIGNAL_HAZARD            = 49    # 0x31
uint16 VEHICLE_SIGNAL_BRAKE             = 65    # 0x41
uint16 VEHICLE_SIGNAL_REVERSE           = 81    # 0x51

# This field is related to https://en.wikipedia.org/wiki/Traffic_light#Single_aspects
uint16 TRAFFIC_LIGHT                    = 2     # 0x02
uint16 TRAFFIC_LIGHT_RED                = 18    # 0x12
uint16 TRAFFIC_LIGHT_YELLOW             = 34    # 0x22
uint16 TRAFFIC_LIGHT_GREEN              = 50    # 0x32
uint16 TRAFFIC_LIGHT_GREEN_LEFT_TURN    = 66    # 0x42
uint16 TRAFFIC_LIGHT_GREEN_RIGHT_TURN   = 66    # 0x42

# Confidence of the signal detection
float32 score
