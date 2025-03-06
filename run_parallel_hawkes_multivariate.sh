# # !/bin/bash
# DEVICE=1
# echo "Running on GPU $DEVICE"
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=1 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
CUDA_VISIBLE_DEVICES=0 python syndata.py -run=1 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=1 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=1 -mode=test -ts_representation_mode=hawkes_multivariate
# # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=1 -mode=privacy -ts_representation_mode=hawkes_multivariat
