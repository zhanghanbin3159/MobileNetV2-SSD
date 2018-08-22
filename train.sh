#!/bin/sh
if ! test -f example/V1_2categroy/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot_V1
../../build/tools/caffe train -solver="solver_train.prototxt" -gpu 0
#-weights="snapshot_V1/0818_with_new_mobilenet_iter_39000.caffemodel"

