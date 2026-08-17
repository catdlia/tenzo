include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // <--- ОСЬ ЦЬОГО НЕ ВИСТАЧАЛО

// --- 1. Опис Діалекту ---
def Tenzo_Dialect : Dialect {
  let name = "tenzo";
  let cppNamespace = "::tenzo";
  let summary = "A custom dialect for deep learning on i3-1215U";
  let description = [{
    Tenzo is a high-level dialect for tensor operations.
  }];
}

// --- 2. Базовий клас ---
class Tenzo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tenzo_Dialect, mnemonic, traits>;

// --- 3. Операція Add ---
// Тепер SameOperandsAndResultType буде знайдено
def AddOp : Tenzo_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise addition";
  let description = [{
    Returns the sum of two tensors of the same shape.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  // Використовуємо functional-type, щоб уникнути проблем з парсингом
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ReluOp : Tenzo_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Rectified Linear Unit activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
}

def SiLuOp : Tenzo_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "Sigmoid Linear Unit (SiLU / Swish) activation";
  let description = [{
    Computes element-wise: silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
    Used in LLaMA/BitNet SwiGLU FFN.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
}

def Relu2Op : Tenzo_Op<"relu2", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU squared activation";
  let description = [{
    Computes element-wise: relu2(x) = max(0, x)^2.
    Used in BitNet architecture.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
}

def MulOp : Tenzo_Op<"mul", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise multiplication";
  let description = [{
    Returns the element-wise product of two tensors of the same shape.
  }];
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def FusedAddReluOp : Tenzo_Op<"fused_add_relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Fused addition and ReLU activation";
  let description = [{
    Performs element-wise addition followed by ReLU: result = max(0, lhs + rhs).
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

// --- Conv2D Operation ---
def Conv2DOp : Tenzo_Op<"conv2d", [Pure]> {
  let summary = "2D Convolution operation";
  let description = [{
    Performs 2D convolution on NHWC tensors with stride=1, no padding.
    Input: [batch, height, width, in_channels]
    Filter: [kernel_h, kernel_w, in_channels, out_channels]
    Output: [batch, out_height, out_width, out_channels]
  }];

  let arguments = (ins
    AnyTensor:$input,   // NHWC format
    AnyTensor:$filter   // HWIO format
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` functional-type(operands, results)
  }];
}

// --- MatMul Operation ---
def MatMulOp : Tenzo_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication C = A * B.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def MatMulQ8Op : Tenzo_Op<"matmul_q8", [Pure]> {
  let summary = "Quantized Matrix multiplication (i8 weights)";
  let description = [{
    Performs matrix multiplication C = A * B_q * scale.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs_q, AnyTensor:$rhs_scales);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs_q `,` $rhs_scales attr-dict `:` functional-type(operands, results)";
}


// --- Quantization Operations ---
def QuantizeOp : Tenzo_Op<"quantize", [Pure]> {
  let summary = "Quantize a tensor to lower precision";
  let description = [{
    Quantizes a floating-point tensor to a lower precision tensor.
    Supports parameterized bit_width (32, 16, 8, 4, 3, 2) and quant_scheme.
    Supported schemes:
    - "fp32", "fp16": No-op or downcast
    - "int8", "int4": Standard integer quantization
    - "classic_tl1": Bitnet.cpp ELUT (2 bits/weight, g=2)
    - "classic_tl2": Bitnet.cpp ELUT with mirror consolidation (1.67 bits/weight, g=3)
    - "experimental_fused": Tenzo native Fused KV-cache compression and implicit shuffle
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    I32Attr:$bit_width,
    StrAttr:$quant_scheme
  );
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

def DequantizeOp : Tenzo_Op<"dequantize", [Pure]> {
  let summary = "Dequantize a tensor to floating point";
  let description = [{
    Dequantizes a lower precision tensor back to a floating-point tensor.
    Supports parameterized bit_width and quant_scheme corresponding to QuantizeOp.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    Optional<AnyType>:$scale,
    I32Attr:$bit_width,
    StrAttr:$quant_scheme
  );
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input (`,` $scale^)? attr-dict `:` functional-type(operands, results)";
}

// --- Attention & KV Cache Operations ---
def AttentionOp : Tenzo_Op<"attention", [Pure]> {
  let summary = "Scaled Dot-Product Attention";
  let description = [{
    Computes attention over Q, K, V tensors.
  }];
  
  let arguments = (ins 
    AnyTensor:$query,
    AnyTensor:$key,
    AnyTensor:$value,
    AnyTensor:$seq_pos
  );
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$query `,` $key `,` $value `,` $seq_pos attr-dict `:` functional-type(operands, results)";
}

def RopeOp : Tenzo_Op<"rope", [Pure]> {
  let summary = "Rotary Position Embedding";
  let description = [{
    Applies Rotary Position Embedding (RoPE) to Query or Key tensor at seq_pos.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$seq_pos
  );
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input `,` $seq_pos attr-dict `:` functional-type(operands, results)";
}

def KVCacheUpdateOp : Tenzo_Op<"kv_cache_update", []> {
  let summary = "Update KV Cache";
  let description = [{
    Appends new key and value tensors to the existing cache.
  }];
  
  let arguments = (ins 
    AnyTensor:$cache_k,
    AnyTensor:$cache_v,
    AnyTensor:$new_k,
    AnyTensor:$new_v,
    AnyTensor:$seq_pos
  );
  let results = (outs 
    AnyTensor:$updated_cache_k,
    AnyTensor:$updated_cache_v
  );
  
  let assemblyFormat = "$cache_k `,` $cache_v `,` $new_k `,` $new_v `,` $seq_pos attr-dict `:` functional-type(operands, results)";
}

def RMSNormOp : Tenzo_Op<"rmsnorm", [Pure]> {
  let summary = "Root Mean Square Normalization";
  let description = [{
    Applies RMSNorm normalization over the last dimension of input tensor using weight.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    AnyTensor:$weight,
    F32Attr:$eps
  );
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input `,` $weight attr-dict `:` functional-type(operands, results)";
}

def EmbeddingOp : Tenzo_Op<"embedding", [Pure]> {
  let summary = "Token Embedding Lookup";
  let description = [{
    Looks up token IDs from input in the embedding weight table.
  }];

  let arguments = (ins
    AnyTensor:$indices,
    AnyTensor:$weight_q,
    Optional<AnyTensor>:$scales
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$indices `,` $weight_q (`,` $scales^)? attr-dict `:` functional-type(operands, results)";
}

// --- ELUT BitLinear Operation (Microsoft TL1 lookup-table approach) ---
def BitLinearElutOp : Tenzo_Op<"bitlinear_elut", [Pure]> {
  let summary = "ELUT (Element-wise Lookup Table) BitLinear matmul";
  let description = [{
    Computes a BitNet 1.58-bit (ternary {-1,0,1}) linear layer using the ELUT
    TL1 approach with group size g=2, *without* materializing a dequantized
    weight matrix.

    Operands:
      input   : FP32 activations [..., K]
      weights : packed uint8 weights [N, K/4]. Each byte holds two 4-bit
                pair-indices; index = (w1+1)*3 + (w2+1) for each consecutive
                ternary weight pair along the K (reduction) axis. The low nibble
                is the first pair of the byte, the high nibble the second.
      scale   : optional FP32 per-tensor dequantization scale.

    The lowering precomputes, for every activation pair (x[2p], x[2p+1]),
    the 3^2 = 9 possible ternary sum combinations a1*w1 + a2*w2 and gathers
    them using the packed 4-bit indices (see LinalgLowering.cpp).
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    Optional<AnyType>:$scale,
    DefaultValuedAttr<I32Attr, "2">:$bit_width,
    DefaultValuedAttr<StrAttr, "\"ternary\"">:$quant_scheme
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights (`,` $scale^)? attr-dict `:` functional-type(operands, results)";
}

// --- TL1 BitLinear Operation (N-major outer product layout for pshufb SIMD) ---
def BitLinearTL1Op : Tenzo_Op<"bitlinear_tl1", [Pure]> {
  let summary = "TL1 (Ternary Lookup Table) BitLinear matmul with N-major layout";
  let description = [{
    Computes a BitNet 1.58-bit linear layer using TL1 (g=2) with packed weights
    in [K/2, N/2] layout for hardware-accelerated pshufb table lookup.

    Operands:
      input   : FP32 activations [..., K]
      weights : packed uint8 weights [K/2, N/2].
      scale   : optional FP32 per-tensor dequantization scale.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    Optional<AnyType>:$scale,
    DefaultValuedAttr<I32Attr, "2">:$bit_width,
    DefaultValuedAttr<StrAttr, "\"ternary\"">:$quant_scheme
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights (`,` $scale^)? attr-dict `:` functional-type(operands, results)";
}

// --- TL1 Pack BitLinear Operation (Microsoft LUT-centric layout) ---
def BitLinearTL1PackOp : Tenzo_Op<"bitlinear_tl1_pack", [Pure]> {
  let summary = "TL1 Pack BitLinear with Microsoft LUT-centric layout";
  let description = [{
    Computes a BitNet 1.58-bit linear layer using Microsoft's TL1 approach.
    Weights are packed in a LUT-centric layout: [N_blocks, K/2, 16 bytes]
    where each byte contains two 4-bit indices for two different N channels.
    This allows parallel table lookups via pshufb.

    Operands:
      input   : FP32 activations [..., K]
      weights : packed uint8 weights [N_blocks, K/2, 16] (16 bytes = 32 indices).
      scale   : optional FP32 per-tensor dequantization scale.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    Optional<AnyType>:$scale,
    DefaultValuedAttr<I32Attr, "2">:$bit_width,
    DefaultValuedAttr<StrAttr, "\"ternary\"">:$quant_scheme
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights (`,` $scale^)? attr-dict `:` functional-type(operands, results)";
}

// --- INT4 BitLinear Operation (4-bit asymmetric/symmetric matmul) ---
def BitLinearINT4Op : Tenzo_Op<"bitlinear_int4", [Pure]> {
  let summary = "INT4 4-bit Quantized Linear matmul";
  let description = [{
    Computes a 4-bit quantized linear layer.
    Weights are packed 2 nibbles per byte: [N, K/2].
    Scale: FP32 per-channel or per-tensor scale [N] or [1].
    Optional zero-point: INT8/FP32 [N] or [1].
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    AnyTensor:$scale,
    Optional<AnyTensor>:$zero_point,
    DefaultValuedAttr<I32Attr, "4">:$bit_width,
    DefaultValuedAttr<StrAttr, "\"int4\"">:$quant_scheme
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights `,` $scale (`,` $zero_point^)? attr-dict `:` functional-type(operands, results)";
}

// --- INT3 BitLinear Operation (3-bit packed matmul) ---
def BitLinearINT3Op : Tenzo_Op<"bitlinear_int3", [Pure]> {
  let summary = "INT3 3-bit Quantized Linear matmul";
  let description = [{
    Computes a 3-bit quantized linear layer.
    Weights are packed 8 weights per 3 bytes: [N, 3*K/8].
    Scale: FP32 per-channel or per-tensor scale [N] or [1].
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    AnyTensor:$scale,
    DefaultValuedAttr<I32Attr, "3">:$bit_width,
    DefaultValuedAttr<StrAttr, "\"int3\"">:$quant_scheme
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights `,` $scale attr-dict `:` functional-type(operands, results)";
}

// --- GGUF BitLinear Operation (llama.cpp Q4_0, Q8_0, K-Quants) ---
def BitLinearGGUFOp : Tenzo_Op<"bitlinear_gguf", [Pure]> {
  let summary = "GGUF Quantized Linear matmul (Q4_0, Q8_0, K-Quants)";
  let description = [{
    Computes a linear layer using GGUF block quantization.
    Q4_0: 32 weights/block (2-byte FP16 scale + 16-byte nibbles)
    Q8_0: 32 weights/block (2-byte FP16 scale + 32-byte int8)
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    AnyTensor:$scales,
    StrAttr:$block_type,
    DefaultValuedAttr<I32Attr, "32">:$block_size
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $weights `,` $scales attr-dict `:` functional-type(operands, results)";
}

// --- GPTQ BitLinear Operation (Groupwise int4 + ActOrder) ---
def BitLinearGPTQOp : Tenzo_Op<"bitlinear_gptq", [Pure]> {
  let summary = "GPTQ Quantized Linear matmul (AutoGPTQ)";
  let description = [{
    Computes a GPTQ quantized linear layer with group size and optional g_idx permutation.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$qweight,
    AnyTensor:$scales,
    AnyTensor:$qzeros,
    Optional<AnyTensor>:$g_idx,
    DefaultValuedAttr<I32Attr, "128">:$group_size
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $qweight `,` $scales `,` $qzeros (`,` $g_idx^)? attr-dict `:` functional-type(operands, results)";
}

// --- AWQ BitLinear Operation (Activation-aware Weight Quantization) ---
def BitLinearAWQOp : Tenzo_Op<"bitlinear_awq", [Pure]> {
  let summary = "AWQ Quantized Linear matmul (AutoAWQ)";
  let description = [{
    Computes an AWQ quantized linear layer with column-interleaved 4-bit packing and group scaling.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$qweight,
    AnyTensor:$scales,
    AnyTensor:$qzeros,
    DefaultValuedAttr<I32Attr, "128">:$group_size
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $qweight `,` $scales `,` $qzeros attr-dict `:` functional-type(operands, results)";
}

// --- EXL2 BitLinear Operation (ExLlamaV2 Variable Bitrate) ---
def BitLinearEXL2Op : Tenzo_Op<"bitlinear_exl2", [Pure]> {
  let summary = "EXL2 Quantized Linear matmul (ExLlamaV2 Variable Bitrate)";
  let description = [{
    Computes an EXL2 quantized linear layer with multi-rate group dispatch.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$qweight,
    AnyTensor:$scales,
    AnyTensor:$qgroups
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input `,` $qweight `,` $scales `,` $qgroups attr-dict `:` functional-type(operands, results)";
}


