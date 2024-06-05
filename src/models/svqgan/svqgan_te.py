from src.models.svqgan.svqgan_base import SVQModel, SVQModelEpoch, SVQModelEpochDVS
import torch
from src.utils.snn_decoder import SnnDecoder
from src.utils.snn_encoder import SnnEncoder
from src.utils.auto_instance import instantiate_from_config
from src.modules.sencoder.base_encoder import EncoderSnn
from src.modules.sdecoder.base_decoder import DecoderSnn
from src.modules.svqvae.quantizer_te import VectorQuantizerSnnTE


class SVQModelTE(SVQModel):

    def __init__(
        self,
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
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
    ):
        super().__init__(ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         n_embed=n_embed,
                         embed_dim=embed_dim)

        # snn encoder and decoder
        self.snn_encoder = SnnEncoder(method=snn_encoder, time_step=time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        self.image_key = image_key
        self.encoder = EncoderSnn(**ddconfig)
        self.decoder = DecoderSnn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizerSnnTE(n_embed,
                                             embed_dim,
                                             beta=0.25,
                                             remap=remap,
                                             sane_index_shape=sane_index_shape,
                                             time_step=time_step)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize",
                                 torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False


class SVQModelTEEpoch(SVQModelEpoch):
    """
    epoch start discriminator
    """

    def __init__(
        self,
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
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
    ):
        super().__init__(ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         n_embed=n_embed,
                         embed_dim=embed_dim)

        # snn encoder and decoder
        self.snn_encoder = SnnEncoder(method=snn_encoder, time_step=time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        self.image_key = image_key
        self.encoder = EncoderSnn(**ddconfig)
        self.decoder = DecoderSnn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizerSnnTE(n_embed,
                                             embed_dim,
                                             beta=0.25,
                                             remap=remap,
                                             sane_index_shape=sane_index_shape,
                                             time_step=time_step)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize",
                                 torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False


class SVQModelTEEpochDVS(SVQModelEpochDVS):
    """
    epoch start discriminator
    """

    def __init__(
        self,
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
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        time_step=3,
        snn_encoder='direct',
        snn_decoder='mean',
    ):
        super().__init__(ddconfig=ddconfig,
                         lossconfig=lossconfig,
                         n_embed=n_embed,
                         embed_dim=embed_dim)

        # snn encoder and decoder
        self.snn_encoder = SnnEncoder(method=snn_encoder, time_step=time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        self.image_key = image_key
        self.encoder = EncoderSnn(**ddconfig)
        self.decoder = DecoderSnn(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizerSnnTE(n_embed,
                                             embed_dim,
                                             beta=0.25,
                                             remap=remap,
                                             sane_index_shape=sane_index_shape,
                                             time_step=time_step)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim,
                                               ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if monitor is not None:
            self.monitor = monitor

        self.automatic_optimization = False
