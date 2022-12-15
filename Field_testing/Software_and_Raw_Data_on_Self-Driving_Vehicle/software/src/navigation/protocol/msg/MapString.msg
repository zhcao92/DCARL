# This message is used for loading geologic map from string content

Header header

# Map type enum
uint8 map_type
uint8 MAP_UNKNOWN = 0
uint8 MAP_OPENDRIVE = 1
uint8 MAP_SUMO = 2

# Map content. It can be encoded, depending on the map_type
string content
