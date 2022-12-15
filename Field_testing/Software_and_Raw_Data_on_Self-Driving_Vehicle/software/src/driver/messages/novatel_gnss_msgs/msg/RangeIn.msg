#Satellite PRN number of range measurement Refer to PRN Numbers on page 48
  uint16 PRN              

#(GLONASS Frequency + 7) (see GLONASS Slot and Frequency Numbers section of this manual)
  uint16 glofreq        

#Pseudorange measurement (m)
  float64 psr         

#Pseudorange measurement standard deviation(m)  
  float32 psr_standard         

#Carrier phase, in cycles (accumulated Doppler range)
  float64 adr            

#Estimated carrier phase standard deviation(cycles)
  float32 adr_standard       

#Instantaneous carrier Doppler frequency (Hz)
  float32 dopp          

#Carrier to noise density ratio C/No = 10[log10(S/N 0 )] (dB-Hz)
  float32 CNo       

#Number of seconds of continuous tracking (no cycle slipping)
  float32 locktime        

#Tracking status
  uint32 track_status        