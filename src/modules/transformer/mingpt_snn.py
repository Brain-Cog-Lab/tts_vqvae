import torch
import torch.nn as nn
from braincog.base.node import LIFNode
import math
import logging
from torch.nn import functional as F
from src.utils.snn_decoder import SnnDecoder
from src.utils.snn_encoder import SnnEncoder

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttentionSnn(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    SNN version
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.k_lif = LIFNode()
        self.q_lif = LIFNode()
        self.v_lif = LIFNode()
        self.key = nn.Sequential(nn.Linear(config.n_embd, config.n_embd),
                                 self.k_lif)
        self.query = nn.Sequential(nn.Linear(config.n_embd, config.n_embd),
                                   self.q_lif)
        self.value = nn.Sequential(nn.Linear(config.n_embd, config.n_embd),
                                   self.v_lif)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head,
                             C // self.n_head).transpose(1,
                                                         2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head,
                               C // self.n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head,
                               C // self.n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float(
                0))  # if no softmax after, then fill 0. Otherwise, fill -inf

        # att = F.softmax(att, dim=-1)  # no softmax for SNN attention
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(
            B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present  # TODO: check that this does not break anything


class BlockSnn(nn.Module):
    """ an unassuming Transformer block, SNN version"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionSnn(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            LIFNode(),  # replace GELU with LIF
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        # TODO: check that training still works
        if return_present: assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class SnnGPT(nn.Module):
    """
    SNN version mini GPT
    """

    def __init__(self,
                 vocab_size,
                 block_size,
                 n_layer=12,
                 n_head=8,
                 n_embd=256,
                 embd_pdrop=0.,
                 resid_pdrop=0.,
                 attn_pdrop=0.,
                 n_unmasked=0,
                 time_step=4,
                 snn_encoder='direct',
                 snn_decoder='mean'):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size,
                           block_size=block_size,
                           embd_pdrop=embd_pdrop,
                           resid_pdrop=resid_pdrop,
                           attn_pdrop=attn_pdrop,
                           n_layer=n_layer,
                           n_head=n_head,
                           n_embd=n_embd,
                           n_unmasked=n_unmasked)

        # snn settings
        self.time_step = time_step
        self.snn_encoder = SnnEncoder(method=snn_encoder,
                                      time_step=self.time_step)
        self.snn_decoder = SnnDecoder(method=snn_decoder)

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[BlockSnn(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e",
                    sum(p.numel() for p in self.parameters()))

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'n_reset'):
                m.n_reset()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        token_embeddings = self.tok_emb(idx)
        token_embeddings_T = self.snn_encoder(token_embeddings)  # (T,...)
        logits_T = []
        for token_embeddings_t in token_embeddings_T:
            logits_t, loss_t = self.forward_step(
                token_embeddings=token_embeddings_t,
                embeddings=embeddings,
                targets=targets)
            # print(logits_t)
            logits_T.append(logits_t)
        logits_T = torch.stack(logits_T, dim=0)
        logits = self.snn_decoder(logits_T)
        loss = None

        self.reset()
        # print(logits)

        return logits, loss

    def forward_step(self, token_embeddings, embeddings=None, targets=None):
        # forward the GPT model

        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :
                                           t, :]  # each position maps to a (learnable) vector
        # position_embeddings nan !!!
        x = self.drop(token_embeddings + position_embeddings)
        # nan here
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))

        return logits, loss

    def forward_with_past(self,
                          idx,
                          embeddings=None,
                          targets=None,
                          past=None,
                          past_length=None):
        token_embeddings = self.tok_emb(idx)

        token_embeddings_T = self.snn_encoder(token_embeddings)  # (T,...)
        logits_T, loss_T, presents_T = [], [], []
        for token_embeddings_t in token_embeddings_T:
            logits_t, loss_t, presents_t = self.forward_with_past_step(
                idx=idx,
                token_embeddings=token_embeddings_t,
                embeddings=embeddings,
                targets=targets,
                past=past,
                past_length=past_length)
            loss_T.append(loss_t)
            logits_T.append(logits_t)
            presents_T.append(presents_t)
        # loss_T = torch.tensor(loss_T)
        logits_T = torch.stack(logits_T, dim=0)
        presents_T = torch.stack(presents_T, dim=0)
        logits = self.snn_decoder(logits_T)
        loss = None
        presents = self.snn_decoder(presents_T)

        self.reset()

        return logits, loss, presents

    def forward_with_past_step(self,
                               idx,
                               token_embeddings,
                               embeddings=None,
                               targets=None,
                               past=None,
                               past_length=None):
        # inference only
        assert not self.training
        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past,
                             dim=-2)  # n_layer, 2, b, nh, len_past, dim_head
            past_shape = list(past.shape)
            expected_shape = [
                self.config.n_layer, 2, idx.shape[0], self.config.n_head,
                past_length, self.config.n_embd // self.config.n_head
            ]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:,
                                               past_length, :]  # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.
                                               shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []  # accumulate over layers
        for i, block in enumerate(self.blocks):
            x, present = block(
                x,
                layer_past=past[i, ...] if past is not None else None,
                return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))

        return logits, loss, torch.stack(
            presents)  # _, _, n_layer, 2, b, nh, 1, dim_head
