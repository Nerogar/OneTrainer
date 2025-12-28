import torch


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    Factorizes the provided number into the product of two numbers.
    Copied from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py#L11
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale=1.0):
    """
    Kronecker product of two tensors.
    """
    if len(w2.shape) == 4: # For Conv2d
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    return rebuild * scale


def rebuild_tucker(t, wa, wb):
    """
    Rebuilds tensor from Tucker decomposition for convolutional layers.
    t: [r, r, k1, k2]
    wa: [r, b]
    wb: [r, d]
    rebuild: [b, d, k1, k2]
    """
    rebuild = torch.einsum('i j k l, i p, j r -> p r k l', t, wa, wb) # [c, d, k1, k2]
    return rebuild
