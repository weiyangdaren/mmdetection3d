# instantiate_model.py
# from mmcv import Config
from mmengine import Config
from mmdet3d.registry import MODELS


def main():
    # 加载配置文件
    cfg = Config.fromfile('projects/OmniDet/configs/base_config.py')

    # 实例化模型
    model = MODELS.build(cfg.model)
    print(model)

if __name__ == '__main__':
    main()
