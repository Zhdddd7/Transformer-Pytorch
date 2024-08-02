import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import einops

class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super(PositionEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype= torch.float).unsqueeze(1)
        div_term = torch.arange(0, max_seq_len, 2).float() * (-math.log(10000.0)/d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

class Attention(nn.Module):
    def __init__(self, d_model,heads = 8, drop_out = 0.1):
        super(Attention, self).__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.heads = heads
        self.dim_head = d_model//heads
        # the output dim after concat of heads
        _dim = self.dim_head * heads
        self.W_0 = nn.Linear(_dim, d_model)
    
    def forward(self, input_Q, input_K, input_V, mask = None):
        q = einops.rearrange(self.W_q(input_Q),'bv(dh) -> bhvd', h = self.heads)
        k = einops.rearrange(self.W_k(input_K),'bv(dh) -> bhvd', h = self.heads)
        v = einops.rearrange(self.W_v(input_V),'bv(dh) -> bhvd', h = self.heads)
        scaler = self.dim_head ** (-0.5)
        scaled_product = torch.einsum('bhid, bhjd -> bhij',q, k) * scaler
        # attention without mask in encoder
        # attention with mask in decoder
        if mask is not None:
            assert mask.shape == scaled_product[2:].shape
            scaled_product = scaled_product.masked_fill(mask, -np.inf)

        attention_score = torch.softmax(scaled_product, dim = -1)
        out = torch.einsum('bhij, bhjd -> bhid', attention_score, v)
        # concat the heads
        out = einops.rearrange(out, 'bhvd -> bv(hd)')
        return self.W_0(out)

class Norm(nn.Module):
    def __init__(self, d_model):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, residual):    
        out = input + residual
        return self.norm(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim = 2048, dropout = 0.1) -> None:
        super(FeedForward, self).__init__()
        self.L1 = nn.Linear(d_model, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(F.relu(self.L1(input)))
        out = self.L2(hidden)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads = 8,dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.attention = Attention(d_model, heads)
        self.addNorm1 = Norm(d_model)
        self.addNorm2 = Norm(d_model)
        self.feedFwd = FeedForward(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input):
        attention_out = self.attention(input, input, input)
        out_1 = self.addNorm1(input, self.dropout1(attention_out))
        out_2 = self.feedFwd(out_1)
        out = self.addNorm2(out_1, self.dropout2(out_2))
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads = 8, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.mask_attention = Attention(d_model, heads)
        self.attention = Attention(d_model, heads)
        self.addNorm1 = Norm(d_model)
        self.addNorm2 = Norm(d_model)
        self.addNorm3 = Norm(d_model)
        self.feedFwd = FeedForward(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, input, encoder_out, mask):
        attn1_out = self.mask_attention(input, input, input, mask)
        out1 = self.addNorm1(input, self.dropout1(attn1_out))
        # without mask in the attention layer2
        attn2_out = self.attention(encoder_out, encoder_out, out1)
        out2 = self.addNorm2(out1, self.dropout2(attn2_out))
        ff_out = self.feedFwd(out2)
        out = self.addNorm3(out2, self.dropout3(ff_out))
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads = 8):
        super(Encoder, self).__init__()
        self.embLayer = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionEncoder(d_model)
        self.encodeLayers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])

    def forward(self, enc_input):
        emb_out = self.embLayer(enc_input)
        out = self.pos_encoder(emb_out)
        for layer in self.encodeLayers:
            out = layer(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads = 8) -> None:
        super(Decoder, self).__init__() 
        self.embLayer = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionEncoder(d_model)
        self.decodeLayers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(N)])
    
    def forward(self, dec_input, enc_output, mask):
        emb_out = self.embLayer(dec_input)
        out = self.pos_encoder(emb_out)
        for layer in self.decodeLayers:
            out = layer(out, enc_output, mask)
        return out
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads = 8):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.linear  = nn.Linear(d_model, trg_vocab)

    def forward(self, enc_input, dec_input, mask):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output, mask)
        output = self.linear(dec_output)
        return output