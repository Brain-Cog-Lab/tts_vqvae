import os
import sys

sys.path.append(os.getcwd())

from src.utils.parser import get_parser_2
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from src.utils.auto_instance import instantiate_from_config

if __name__ == "__main__":
    # parse
    parser = get_parser_2()
    args = parser.parse_args()

    # load config
    config = OmegaConf.load(args.base[0])

    # get save dir
    save_dir = os.path.join('./res', config.exp.name, config.exp.index,
                            'samples')
    os.makedirs(save_dir, exist_ok=True)

    # get unfold dir for dvs
    unfold_dir = None
    if config.sample_params.time_unfold:
        unfold_dir = os.path.join('./res', config.exp.name, config.exp.index,
                                  'unfold')
        os.makedirs(unfold_dir, exist_ok=True)

    # seed
    seed_everything(config.exp.seed)

    # sampler
    sampler = instantiate_from_config(config.sampler)

    # sample
    sampler.save_sample_unconditional_te_dvs(**config.sample_params,
                                             save_dir=save_dir,
                                             unfold_dir=unfold_dir)
