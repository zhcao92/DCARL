# Describing the size object in 2D space (in pixels) with uncertainty

float64 length_x # width
float64 length_y # length

# Row-major representation of the 2x2 covariance matrix
# In order, the parameters are: (length_x, length_y)
float64[4] covariance
