inp=/home/muhammed/projects/metric_hcm/results/EMDB_2/P2/20_outdoor_walk

python demo.py --imagedir $inp/rgb \
    --calib $inp/cam_intrinsics.txt \
    --stride=1 \
    --reconstruction_path $inp/droid_masked/filterdepth \
    --maskdir $inp/mask \
    --depthdir $inp/monodepth/zoedepth \
    --disable_vis \
    --filter_inp_depth


# python demo.py --imagedir $inp/rgb \
#     --calib $inp/cam_intrinsics.txt \
#     --stride=1 \
#     --reconstruction_path $inp/droid_masked/metric3d_vit_giant2_filtered \
#     --maskdir $inp/mask \
#     --depthdir $inp/monodepth/metric3d_vit_giant2 \
#     --disable_vis \
#     --filter_inp_depth