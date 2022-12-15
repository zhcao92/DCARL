# Package Summary
This package contains filters on detection results with lidar point cloud.

# Node `bbox`
This node visualize bounding box detection results

## Input Topic
|Topic|Condition|Type|Description|
|---|---|---|---|
|`objects_detected`|`visualize_type` == `DetectionBoxArray`|`zzz_perception_msgs/DetectionBoxArray`|Input detection result|
## Output Topic
|Topic|Condition|Type|Description|
|---|---|---|---|
|`objects_visual`||`visualization_msgs/MarkerArray`|Detected objects|
## Parameters
|Parameter|Type|Description|Default|
|---|---|---|---|
|`visualize_type`|string|Visualized message type|`DetectionBoxArray`|
|`marker_lifetime`|float|Duration of generated markers|`0.1`|
|`box_color`|float[4]|Color of boxes (RGBA)|`51.,128.,204.,0.8`|
|`centroid_color`|float[4]|Color of centroids (RGBA)|`77.,121.,255.,0.8`|
|`centroid_scale`|float|Scale of centroid markers|`0.5`|
|`label_color`|float[4]|Color of labels (RGBA)|`255.,255.,255.,1.0`|
|`label_scale`|float|Scale of label markers|`0.5`|
|`label_height`|float|Elevation of label markers|`1.0`|
|`box_max_size`|float|Euclidean Clustering distance threshold|`10`|
