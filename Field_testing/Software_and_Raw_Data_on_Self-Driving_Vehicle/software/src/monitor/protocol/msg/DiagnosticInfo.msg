# This message contains container for diagnostic information that can be logged or displayed.


# Time stamp for synchronization
time time

# Data type of the diagnostic data
uint8 type
uint8 TYPE_JSON_KV = 0 # Default message, contains only key-value pairs
uint8 TYPE_MSGPACK_KV = 0 # message encoded in msgpack, contains only key-value pairs

# The diagnostic data to be sent
string data
