import torch


def precondition_lora_grad(p: torch.Tensor, grad: torch.Tensor, delta: float = 1e-6) -> torch.Tensor:
    """
    Applies Riemannian preconditioning to LoRA gradients as described in
    "Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models" (arXiv:2402.02347v3).
    """
    pair = getattr(p, '_lora_pair', None)
    if pair is None:
        return grad

    is_lora_B = getattr(p, '_is_lora_B', False)
    is_lora_A = getattr(p, '_is_lora_A', False)

    if not (is_lora_A or is_lora_B) or p.norm() < 1e-6:
        return grad

    if is_lora_B:
        # p is B [d_out, rank], pair is A [rank, d_in]
        d_out = p.shape[0]
        rank = p.numel() // d_out

        A = pair.detach().view(rank, -1).float()
        grad_f = grad.view(d_out, rank).float()

        # Preconditioner for B: (A @ A.T + delta * I)^-1
        AAt = A @ A.mT
        eye = torch.eye(rank, device=p.device, dtype=torch.float32)

        # dB_scaled = dB @ (A @ A.T)^-1
        grad_scaled = torch.linalg.solve(AAt + delta * eye, grad_f.mT).mT

    else:
        # p is A [rank, d_in], pair is B [d_out, rank]
        rank = p.shape[0]
        B = pair.detach().view(-1, rank).float()
        grad_f = grad.view(rank, -1).float()

        # Preconditioner for A: (B.T @ B + delta * I)^-1
        BtB = B.mT @ B
        eye = torch.eye(rank, device=p.device, dtype=torch.float32)

        # dA_scaled = (B.T @ B)^-1 @ dA
        grad_scaled = torch.linalg.solve(BtB + delta * eye, grad_f)

    # Return the scaled gradient in the original shape and dtype
    return grad_scaled.view_as(grad).to(grad.dtype)
