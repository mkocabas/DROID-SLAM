# inp=/home/muhammed/projects/metric_hcm/results/EMDB_2/P2/24_outdoor_long_walk
inp=/home/muhammed/projects/metric_hcm/results/EMDB_2/P3/28_outdoor_walk_lunges

python demo.py --imagedir $inp/rgb \
    --calib $inp/cam_intrinsics.txt \
    --stride=1 \
    --reconstruction_path $inp/droid_masked/filterdepth_huber \
    --maskdir $inp/mask \
    --depthdir $inp/monodepth/metric3d_vit_large \
    --disable_vis \
    --filter_inp_depth \
    --disp_residual_kernel huber


# python demo.py --imagedir $inp/rgb \
#     --calib $inp/cam_intrinsics.txt \
#     --stride=1 \
#     --reconstruction_path $inp/droid_masked/metric3d_vit_giant2_filtered \
#     --maskdir $inp/mask \
#     --depthdir $inp/monodepth/metric3d_vit_giant2 \
#     --disable_vis \
#     --filter_inp_depth