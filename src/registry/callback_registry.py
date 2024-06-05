from pytorch_lightning.callbacks import ModelCheckpoint
from src.callbacks.image_logger import ImageLoggerEpoch, ImageLoggerBatch, ImageLoggerEpochDVS
from src.callbacks.when_stop import StopBatch
from src.callbacks.setup_callback import SetupCallback2


def stop_batch(from_config, args: dict):
    if not from_config:
        from_config = dict()
    return StopBatch(**from_config)


def model_ckpt(from_config, args: dict):
    if not from_config:
        from_config = dict()
    dirpath = args['ckpt_dir']
    filename = '{epoch}-{step}'
    verbose = True
    save_last = True
    return ModelCheckpoint(**from_config,
                           dirpath=dirpath,
                           filename=filename,
                           verbose=verbose,
                           save_last=save_last)


def image_logger_epoch(from_config, args: dict):
    if not from_config:
        from_config = dict()
    img_dir = args['img_dir']
    return ImageLoggerEpoch(**from_config, img_dir=img_dir)


def image_logger_epoch_dvs(from_config, args: dict):
    if not from_config:
        from_config = dict()
    img_dir = args['img_dir']
    return ImageLoggerEpochDVS(**from_config, img_dir=img_dir)


def image_logger_batch(from_config, args: dict):
    if not from_config:
        from_config = dict()
    img_dir = args['img_dir']
    return ImageLoggerBatch(**from_config, img_dir=img_dir)


def setup_fit(from_config, args: dict):
    if not from_config:
        from_config = dict()
    img_dir = args['img_dir']
    res_dir = args['res_dir']
    ckpt_dir = args['ckpt_dir']
    tb_dir = args['tb_dir']
    return SetupCallback2(**from_config,
                          img_dir=img_dir,
                          res_dir=res_dir,
                          ckpt_dir=ckpt_dir,
                          tb_dir=tb_dir)
