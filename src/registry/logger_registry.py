from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


def csv_logger(from_config, args: dict):
    if not from_config:
        from_config = dict()
    save_dir = args['res_dir']
    name = 'pl_csv_log'
    return CSVLogger(**from_config, save_dir=save_dir, name=name)


def tb_logger(from_config, args: dict):
    if not from_config:
        from_config = dict()
    save_dir = args['tb_dir']
    name = 'pl_tb_log'
    return TensorBoardLogger(**from_config, save_dir=save_dir, name=name)
