  uint8[3] sync

  uint8 header_length

  uint16 message_id

#MessageType no need up to now
  uint8 message_type

# Address of the data port the log was received on.
  uint8 port_address  

# Message length (not including the header nor the CRC).
  uint16 message_length  # Message length (not including the header nor the CRC).

# Counts down from N-1 to 0 for multiple related logs.
  uint16 sequence   

# Time the processor was idle in last second between logs with same ID.
  uint8 idle  

# Indicates the quality of the GPS time.
  uint8 time_status

# GPS Week number.
  uint16 gps_week       

# Milliseconds of week.
  uint32 gps_millisecs  

# Receiver status.
  uint32 status         

  uint16 reserved

# Receiver software build number.
  uint16 version  