inp=/home/muhammed/projects/metric_hcm/results/EMDB_2/P2/20_outdoor_walk

python demo.py --imagedir $inp/rgb \
    --calib $inp/cam_intrinsics.txt \
    --stride=1 \
    --reconstruction_path $inp/droid_masked/filterdepth \
    --maskdir $inp/mask \
    --depthdir $inp/monodepth/metric3d_vit_large \
    --disable_vis
    