# Run demo
```bash
python demo/pcd_demo.py demo/data/kitti/000008.bin configs/my_cfgs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
```

# Prepare data
```bash
python tools/create_data.py nuscenes --root-path data/nuscenes --version v1.0-mini --extra-tag mini --workers 32
python projects/OmniDet/tools/omni_converter.py
```

# Train
```bash
python tools/train.py projects/OmniDet/configs/base_config.py
python tools/train.py projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py
```

# Test
```bash
python tools/test.py projects/OmniDet/configs/base_config.py work_dirs/base_config/epoch_14.pth
python tools/test.py projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py work_dirs/petr_vovnet_gridmask_p4_800x320/epoch_24.pth
```


# visual
```bash
python tools/test.py projects/OmniDet/configs/base_config.py work_dirs/base_config/epoch_20.pth --show --show-dir work_dirs --task multi-view_det

python tools/misc/visualize_results.py projects/OmniDet/configs/base_config.py --result ${RESULTS_PATH} --show-dir ${SHOW_DIR}

python tools/misc/browse_dataset.py projects/OmniDet/configs/base_config.py --task lidar_det --output-dir work_dirs

```