#!/bin/bash
parent_path=$(cd `dirname $0`; pwd)
project_root_path=$(cd `dirname $parent_path`; pwd)
# export MODEL_ROOT=$project_root_path/model

cat>$project_root_path/scripts/cnnseg_trt_builder.yaml<<EOF
gModelType: caffe
gDataType: 0
gMaxBatchSize: 1
gMaxWorkspaceSize: 30
gInputNode: [input]
gCaffePrototxt: $project_root_path/models/cnnseg/deploy.prototxt
gCaffeModel: $project_root_path/models/cnnseg/deploy.caffemodel
gOutputNode: [deconv0]
gCnnSeg: True
EOF

source $project_root_path/devel/setup.bash
rosrun trt trt_builder $project_root_path/scripts/cnnseg_trt_builder.yaml
rm $project_root_path/scripts/cnnseg_trt_builder.yaml
# mv $project_root_path/models/cnnseg/deploy.caffemodel_FP32.trt $project_root_path/models/cnnseg/cnnseg.640.FP32.x64.trt