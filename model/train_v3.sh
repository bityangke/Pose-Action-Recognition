#!/usr/bin/env sh
/vision/u/liyues/software/caffe/build/tools/caffe train --solver=train_v3_solver.prototxt --gpu=$1 --weights=../model/_trained_COCO/pose_iter_440000.caffemodel 2>&1 | tee ./output_train_v3.txt
