# This message is used to report module status. This message is only reported when changes happen.

Header header

uint8 module_category

uint8 MODULE_PERCEPTION_LIDAR = 1
uint8 MODULE_PERCEPTION_CAMERA = 2

string module_method

int8 module_status
int8 STATUS_DISABLED = 0
int8 STATUS_ENABLED = 1
int8 STATUS_FINISHED = 2
int8 STATUS_CRASHED = -1
