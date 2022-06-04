import torch
from performer_pytorch import PerformerEncDec

SRC_SEQ_LEN = 4096
TGT_SEQ_LEN = 4096
GENERATE_LEN = 512

enc_dec = PerformerEncDec(
    dim = 512,
    tie_token_embed = True,
    enc_num_tokens = 20000,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = SRC_SEQ_LEN,
    dec_num_tokens = 20000,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = TGT_SEQ_LEN,
)

src = torch.randint(0, 20000, (1, SRC_SEQ_LEN))
tgt = torch.randint(0, 20000, (1, TGT_SEQ_LEN))
src_mask = torch.ones_like(src).bool()
tgt_mask = torch.ones_like(src).bool()

# train
enc_dec.train()
loss = enc_dec(src, tgt, enc_mask = src_mask, dec_mask = tgt_mask)
loss.backward()

# generate
generate_in = torch.randint(0, 20000, (1, SRC_SEQ_LEN)).long()
generate_out_prime = torch.tensor([[0.]]).long() # prime with <bos> token
samples = enc_dec.generate(generate_in, generate_out_prime, seq_len = GENERATE_LEN, eos_token = 1) # assume 1 is id of stop token
print(samples.shape) # (1, <= GENERATE_LEN) decode the tokens