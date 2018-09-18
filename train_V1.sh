#!/bin/sh
if ! test -f example/V1_4_sp/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot_V1_4
/media/ziwei/Harddisk02/HanBin/TOOL/workspace_caffe/ssd/build/tools/caffe train -solver="solver_train_V1.prototxt" -gpu 0
-weights="snapshot_V1_4/mobilenet_iter_10000.caffemodel"

