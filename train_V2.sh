#!/bin/sh
if ! test -f example/V2/MobileNetSSDV2_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot_V2
../../build/tools/caffe train -solver="solver_train_V2.prototxt" -gpu 0
-weights="snapshot_V2/0816_mobilenet_iter_30000.caffemodel"

