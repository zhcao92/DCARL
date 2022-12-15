# Interface for vehicle control

# ---------- Trottle ----------
# Throttle control by-wire enabled
bool throttle_available
# Throttle control overtaken
bool throttle_override 
# Reported throttle level
float32 throttle_report

# ---------- Brake ----------
# Braking control by-wire enabled
bool brake_available
# Braking control overtaken
bool brake_override
# Reported braking level
float32 brake_report

# ---------- Steering ----------
# Steering control by-wire enabled
bool steer_available
# Steering control overtaken
bool steer_override
# Reported steering level
float32 steer_report
