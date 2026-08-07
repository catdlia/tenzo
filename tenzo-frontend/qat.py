import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        # 1. Знаходимо масштаб (scale), зазвичай середнє абсолютне значення
        scale = weight.abs().mean().clamp(min=1e-5)
        
        # 2. Масштабуємо ваги
        w_scaled = weight / scale
        
        # 3. Квантуємо до {-1, 0, 1}
        w_quantized = torch.round(w_scaled).clamp(-1, 1)
        
        # 4. Де-квантуємо назад (повертаємо масштаб для правильних обчислень активацій)
        w_dequantized = w_quantized * scale
        
        return w_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        # МАГІЯ STE: На зворотному проході ми просто ігноруємо 
        # недиференційовану функцію round() і пропускаємо градієнт далі
        return grad_output


class BitLinear(nn.Linear):
    """
    QAT layer for 1.58-bit (ternary) weights using Straight-Through Estimator.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input):
        # Apply ternary quantization to weights on the forward pass
        quantized_weight = TernaryQuantizeSTE.apply(self.weight)
        return F.linear(input, quantized_weight, self.bias)

