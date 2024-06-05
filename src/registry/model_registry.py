from src.models.vqgan import VQModel
from src.models.svqgan.svqgan_base import SVQModel, SVQModelEpochDVS, SVQModelEpoch
from src.models.svqgan.svqgan_te import SVQModelTE, SVQModelTEEpoch, SVQModelTEEpochDVS
from src.modules.losses.vqperceptual import VQLPIPSWithDiscriminator


def snnte_facehq(
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
):
    # temporal embedding quantizer
    return SVQModelTE(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        ckpt_path=ckpt_path,
        ignore_keys=ignore_keys,
        image_key=image_key,
        colorize_nlabels=colorize_nlabels,
        monitor=monitor,
        remap=remap,
        sane_index_shape=sane_index_shape,
        time_step=time_step,
        snn_encoder=snn_encoder,
        snn_decoder=snn_decoder,
    )


def snnte_lsunbed(
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
):
    # temporal embedding quantizer
    return SVQModelTE(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        ckpt_path=ckpt_path,
        ignore_keys=ignore_keys,
        image_key=image_key,
        colorize_nlabels=colorize_nlabels,
        monitor=monitor,
        remap=remap,
        sane_index_shape=sane_index_shape,
        time_step=time_step,
        snn_encoder=snn_encoder,
        snn_decoder=snn_decoder,
    )


def snnte_epoch(
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
):
    # temporal embedding quantizer
    # epoch start discriminator
    return SVQModelTEEpoch(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        ckpt_path=ckpt_path,
        ignore_keys=ignore_keys,
        image_key=image_key,
        colorize_nlabels=colorize_nlabels,
        monitor=monitor,
        remap=remap,
        sane_index_shape=sane_index_shape,
        time_step=time_step,
        snn_encoder=snn_encoder,
        snn_decoder=snn_decoder,
    )


def dvste_epoch(
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
):
    # temporal embedding quantizer
    # epoch start discriminator
    return SVQModelTEEpochDVS(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim,
        ckpt_path=ckpt_path,
        ignore_keys=ignore_keys,
        image_key=image_key,
        colorize_nlabels=colorize_nlabels,
        monitor=monitor,
        remap=remap,
        sane_index_shape=sane_index_shape,
        time_step=time_step,
        snn_encoder=snn_encoder,
        snn_decoder=snn_decoder,
    )
