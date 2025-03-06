# # !/bin/bash
# DEVICE=1
# echo "Running on GPU $DEVICE"
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=1 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
CUDA_VISIBLE_DEVICES=0 python syndata.py -run=1 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=1 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=1 -mode=test -ts_representation_mode=hawkes_multivariate
# # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=1 -mode=privacy -ts_representation_mode=hawkes_multivariate

# # # CUDA_VISIBLE_DEVICES= python syndata.py -run=2 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=2 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=2 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=2 -mode=test -ts_representation_mode=hawkes_multivariate
# # # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=2 -mode=privacy -ts_representation_mode=hawkes_multivariate

# # # CUDA_VISIBLE_DEVICES= python syndata.py -run=3 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=3 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=3 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=3 -mode=test -ts_representation_mode=hawkes_multivariate
# # # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=3 -mode=privacy -ts_representation_mode=hawkes_multivariate

# # # CUDA_VISIBLE_DEVICES= python syndata.py -run=4 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=4 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=4 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# # CUDA_VISIBLE_DEVICES= python syndata.py -run=4 -mode=test -ts_representation_mode=hawkes_multivariate
# # # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=4 -mode=privacy -ts_representation_mode=hawkes_multivariate

# # # CUDA_VISIBLE_DEVICES= python syndata.py -run=5 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=5 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=5 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=5 -mode=test -ts_representation_mode=hawkes_multivariate
# # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=5 -mode=privacy -ts_representation_mode=hawkes_multivariate

# # CUDA_VISIBLE_DEVICES= python syndata.py -run=6 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=6 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=6 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=6 -mode=test -ts_representation_mode=hawkes_multivariate
# # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=6 -mode=privacy -ts_representation_mode=hawkes_multivariate

# # CUDA_VISIBLE_DEVICES= python syndata.py -run=7 -mode=sanity_test -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=7 -mode=train -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=7 -mode=privacy_ablation -ts_representation_mode=hawkes_multivariate
# CUDA_VISIBLE_DEVICES= python syndata.py -run=7 -mode=test -ts_representation_mode=hawkes_multivariate
# # # CUDA_VISIBLE_DEVICES=$DEVICE python syndata.py -run=7 -mode=privacy -ts_representation_mode=hawkes_multivariate
