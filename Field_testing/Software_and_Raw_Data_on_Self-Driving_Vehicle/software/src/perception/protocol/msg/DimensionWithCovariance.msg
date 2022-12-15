# Describing the size object in 3D space (in meters) with uncertainty

float64 length_x # width
float64 length_y # height
float64 length_z # length

# Row-major representation of the 3x3 covariance matrix
# In order, the parameters are: (length_x, length_y, length_z)
float64[9] covariance