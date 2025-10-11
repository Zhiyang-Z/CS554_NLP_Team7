import torch

def gen_padding_mask(x: torch.tensor, padding_token: int) -> torch.tensor:
    """generate padding mask where padding position filled with True, otherwise, False."""
    return x == padding_token

def gen_padding_mask_for_self_attention(padding_mask: torch.tensor) -> torch.tensor:
    assert padding_mask.ndim == 2 # padding_mask shape: [batch_size, length]
    length = padding_mask.shape[1]
    padding_mask = padding_mask.unsqueeze(1).repeat(1, length, 1) # padding_mask shape: [batch_size, length, length]
    padding_mask = padding_mask | padding_mask.transpose(1, 2)
    """
    padding_mask looks like: [False, False, False, True, True]
                             [False, False, False, True, True]
                             [False, False, False, True, True]
                             [True,  True,  True,  True, True]
                             [True,  True,  True,  True, True]
    """
    return padding_mask # shape: [batch_size, length, length]
    

# test
if __name__ == '__main__':
    print(gen_attn_mask(7))
    padding_mask = gen_padding_mask(torch.tensor([[1,2,3,0], [4,5,0,0]]), 0)
    padding_mask_for_self_att = gen_padding_mask_for_self_attention(padding_mask)
    print(padding_mask_for_self_att)
    import matplotlib.pyplot as plt
    pos_encoding = gen_pos_encoding(100, 768)
    cax = plt.matshow(pos_encoding)
    plt.gcf().colorbar(cax)
    plt.show()