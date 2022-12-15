#Gps_Ephemeris
Header header

#novatel long header
LongHeader long_header

# Satellite prn number
  uint32 prn      

# Time stamp of subframe 0
  float64 tow        

# Health status -a 6-bit health code as defined in ICD-GPS-200
  uint32 health   

# issue of ephemeris data 1
  uint32 iode1    

# issue of ephemeris data 2
  uint32 iode2    

# GPS reference week number
  uint32 week     

# Z count week number
  uint32 z_week   

# reference time for ephemeris, seconds
  float64 toe        

# semi-major axis, metres
  float64 A          

# Mean motion difference, radians/second
  float64 delta_A    

# Mean anomaly of reference time, radians
  float64 M_0        

# Eccentricity. dimensionless-quantity defined for a conic section where e=0 is a circle, 
#e=1 is a parabola 0<e<1 os an ellipse and e>1 is a hyperbola
  float64 ecc        

# Argument of perigee, radians -measurement along the orbital path from the ascending node to 
#the point where the SV os closest to the earth, in the direction of the SV's motion
  float64 omega      

# Argument of latitude
  float64 cuc        

# Argument of latitude
  float64 cus        

# Orbit radius
  float64 crc        

# Orbit radius
  float64 crs

# Inclination
  float64 cic        

# Inclination
  float64 cis        

# Inclination angle at reference time, radians
  float64 I_0        

# Rate of inclination angle, radians/second
  float64 dot_I      
  
# right ascension, radians
  float64 omega_0    

# rate of right ascension, radians/second
  float64 dot_omega  

# issue of data clock
  uint32 iodc     

# SV clock correction term, seconds
  float64 toc        

# Estimated group delay difference seconds
  float64 tgd        

# Clock aging parameter. seconds
  float64 af0        

# Clock aging parameter
  float64 af1        

# Clock aging parameter
  float64 af2        

# Anti-spoofing on : 0=false, 1=true
  uint32 AS       

# Corrected mean motion, radians/second
  float64 N          

# User Range Acceracy variance.
  float64 ura        
