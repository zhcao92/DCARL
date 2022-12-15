#!/bin/bash
parent_path=$(cd `dirname $0`; pwd)
project_root_path=$(cd `dirname $parent_path`; pwd)
# export MODEL_ROOT=$project_root_path/model

cat>$project_root_path/scripts/pointpillars_trt_builder.yaml<<EOF
gModelType: onnx
gDataType: 1
gMaxBatchSize: 1
gMaxWorkspaceSize: 30
gInputNode: ["coors_input_", "voxel_num_", "points"]
gOnnxModel: $project_root_path/models/pointpillars/pointpillars-1.onnx
gOnnxModelTwo: $project_root_path/models/pointpillars/pointpillars-2.onnx
gOutputNode: ["pointpillars_part2/features_car", "pointpillars_part2/features_ped_cyc"]
EOF

source $project_root_path/devel/setup.bash
rosrun trt trt_builder $project_root_path/scripts/pointpillars_trt_builder.yaml
rm $project_root_path/scripts/pointpillars_trt_builder.yaml
# mv $project_root_path/models/cnnseg/deploy.caffemodel_FP32.trt $project_root_path/models/cnnseg/cnnseg.640.FP32.x64.trt
