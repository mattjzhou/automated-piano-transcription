from torch import nn


class Baseline(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.transformer = nn.Transformer(nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          batch_first=True)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, y, src_key_padding_mask=None, tgt_key_padding_mask=None):
        embedded_y = self.embed(y)
        outs = self.transformer(x, embedded_y, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        outs = self.linear(outs)
        return outs
