import sys
import os

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from src.utils.parser import get_parser_2
import argparse
from pytorch_lightning import seed_everything
from src.utils.auto_instance import instantiate_from_config, instantiate_from_config_args
from pytorch_lightning.trainer import Trainer


def get_dir_name(exp_config):
    name = exp_config.name
    index = exp_config.index
    root = './res'
    res_dir = os.path.join(root, name, index)
    ckpt_dir = os.path.join(res_dir, 'ckpt')
    img_dir = os.path.join(res_dir, 'img')
    tb_dir = os.path.join(res_dir, 'tb')
    return res_dir, ckpt_dir, img_dir, tb_dir


def setup_model(model_config):
    return instantiate_from_config(model_config)


def setup_callback(callback_config, callback_args: dict):
    print(callback_config)
    callback_list = []
    for key in callback_config.keys():
        callback_list.append(
            instantiate_from_config_args(callback_config[key], callback_args))
    return callback_list


def setup_logger(logger_config, logger_args: dict):
    logger_list = []
    for key in logger_config.keys():
        logger_list.append(
            instantiate_from_config_args(logger_config[key], logger_args))
    return logger_list


def setup_trainer(trainer_config, trainer_args: dict):
    # setup trainer callback
    callback_args = trainer_args
    callbacks = setup_callback(trainer_config.callbacks,
                               callback_args=callback_args)

    # setup trainer loggers
    logger_args = trainer_args
    logger = setup_logger(trainer_config.loggers, logger_args=logger_args)

    del trainer_config.callbacks
    del trainer_config.loggers

    return Trainer(**trainer_config, callbacks=callbacks, logger=logger)


def setup_lightning(lightning_config, lightning_args: dict):
    # setup trainer
    train_args = lightning_args
    trainer = setup_trainer(lightning_config.trainer, train_args)
    return trainer


def setup_data(data_config):
    data = instantiate_from_config(data_config)
    return data


if __name__ == "__main__":
    # parse
    parser = get_parser_2()
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.base[0])

    # get dir (not make yet)
    res_dir, ckpt_dir, img_dir, tb_dir = get_dir_name(config.exp)

    # seed
    seed_everything(config.exp.seed)

    # setup model
    model = setup_model(config.model)

    # setup lightning (include trainer)
    lightning_args = dict()
    lightning_args['res_dir'] = res_dir
    lightning_args['ckpt_dir'] = ckpt_dir
    lightning_args['img_dir'] = img_dir
    lightning_args['tb_dir'] = tb_dir
    trainer = setup_lightning(config.lightning, lightning_args)

    # setup data
    data = setup_data(config.data)

    # calculate learning rate according t0 golbal batchsize
    accumulate_grad_batches = config.lightning.trainer.accumulate_grad_batches
    batch_size = data.batch_size
    base_lr = config.model.base_learning_rate
    ngpus = len(config.lightning.trainer.devices)
    model.learning_rate = accumulate_grad_batches * ngpus * batch_size * base_lr
    print(f"Setting lr to {model.learning_rate}")

    # fit
    if config.exp.is_resume:
        trainer.fit(model=model,
                    datamodule=data,
                    ckpt_path=config.exp.resume_path)
    else:
        trainer.fit(model=model, datamodule=data)
