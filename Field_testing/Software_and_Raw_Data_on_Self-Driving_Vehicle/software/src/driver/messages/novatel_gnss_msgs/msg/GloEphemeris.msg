#Glo_Ephemeris
Header header

#novatel long header
LongHeader long_header

# Slot information offset-PRNidentification(Slot+37).
  uint16 sloto      

# frequency channel offset for satellite in the range 0 to 20
  uint16 freqo      

# Satellite type where(0=GLO_SAT, 1=GLO_SAT_M, 2=GLO_SAT_K)
  uint8 sat_type    

# Reserved
  uint8 reserved_1  

# reference week of ephemeris(GPS reference time)
  uint16 e_week     

# reference time of ephemeris(GPS reference time) in ms
  uint32 e_time     

# integer seconds between GPS and GLONASS time.
# A positive value implies GLONASS is ahead of GPS reference time.
  uint32 t_offset   

# Calendar number of day within 4 year interval starting at Jan 1 of leap year
  uint16 Nt  

# Reserved
  uint8 reserved_2  

# Reserved
  uint8 reserved_3  

# 15 minute interval number corresponding to ephemeris reference time
  uint32 issue      

# Ephemeris health where 0-3=GOOD, 4-15=BAD
  uint32 health     

# X coordinate for satellite at reference time (PZ-90.02), in meters
  float64 pos_x  

# Y coordinate for satellite at reference time (PZ-90.02), in meters
  float64 pos_y  

# Z coordinate for satellite at reference time (PZ-90.02), in meters
  float64 pos_z  

# X coordinate for satellite velocity at reference  time(PZ-90.02), in meters/s
  float64 vel_x  

# Y coordinate for satellite velocity at reference time(PZ-90.02), in meters/s
  float64 vel_y  

# Z coordinate for satellite velocity at reference time(PZ-90.02), in meters/s
  float64 vel_z  

# X coordinate for lunisolar acceleration at reference time(PZ-90.02), in meters/s/s
  float64 acc_x  

# Y coordinate for lunisolar acceleration at reference time(PZ-90.02), in meters/s/s
  float64 acc_y  

# Z coordinate for lunisolar acceleration at reference time(PZ-90.02), in meters/s/s
  float64 acc_z  

# Correction to the nth satellite time t_n relative to GLONASS time_t, in seconds
  float64 tau_n  

# Time difference between navigation RF signal transmitted in L2 sub-band and
# navigation RF signal transmitted in L1 sub-band by nth satellite , in seconds
  float64 delta_tau_n  

# frequency correction , in seconds/second
  float64 gamma        

# Time of frame start(since start of GLONASS day), in seconds
  uint32 Tk   

# technological parameter
  uint32 P    

# User range
  uint32 Ft   

# age of data, in days
  uint32 age  

# information flags
  uint32 Flags  