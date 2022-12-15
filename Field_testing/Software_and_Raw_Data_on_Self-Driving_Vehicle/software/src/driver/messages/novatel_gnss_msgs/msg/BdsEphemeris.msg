#Bds_Ephemeris
Header header

#novatel long header
LongHeader long_header

# ID/ranging code
  uint32 satellite_id  

# week number
  uint32 week 

# user range accuracy(meters).
# This is the evaluated URAI/URA lookup-table value         
  float64 ura  

# Autonomous satellite health flag.
# 0 means broadcasting satellite is good, 1 means not
  uint32 health1  

# Equipment group delay differential for the B1 signal(seconds)
  float64 tdg1

# Equipment group delay differential for the B2 signal(seconds)
  float64 tdg2  

# Age of data, clock
  uint32 aodc  

# Reference time of clock parameters
  uint32 toc   

# Constant term of clock correction polynomial(seconds)
  float64 a0  

# Linear term of clock correction polynomial (seconds/seconds)
  float64 a1  

# Quadratic term of clock correction polynomial (second/seconds^2)
  float64 a2  

# Age of data, ephemeris
  uint32 aode    

# Reference time of ephemeris parameters
  uint32 toe      

# Square root of semi-major axis (sqrt(meters))
  float64 rootA      

# Eccentricity (sqrt(meters))
  float64 ecc        

# Argument of perigee
  float64 omega      

# Mean motion difference from computed value (radians/second)
  float64 delta_N    

# Mean anomaly at reference time (radians)
  float64 M0         

# Longitude of ascending node of orbital of plane computed according to reference time(radians)
  float64 omega0     

# Rate of right ascension(radians/second)
  float64 rra        

# inclination angle at reference time(radians)
  float64 inc_angle  

# Rate of inclination angle(radians/second)
  float64 idot       

# Amplitude of cosine harmonic correction term to the argument of latitude(radians)
  float64 cuc  

# Amplitude of sine harmonic correction term to the argument of latitude(radians)
  float64 cus  

# Amplitude of cosine harmonic correction term to the orbit radius(meters)
  float64 crc  

# Amplitude of sine harmonic correction term to the orbit radius(meters)
  float64 crs  

# Amplitude of cosine harmonic correction term to the angle of inclination(radians)
  float64 cic  

# Amplitude of sine harmonic correction term to the angle of inclination(radians)
  float64 cis  
