#!/bin/sh
latest=snapshot_V1/0822_without_new_mobilenet_iter_30000.caffemodel
#latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
../../build/tools/caffe train -solver="solver_test_V1.prototxt" \
--weights=$latest \
-gpu 0
