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

// --- Quantization Operations ---
def QuantizeOp : Tenzo_Op<"quantize", [Pure]> {
  let summary = "Quantize a tensor to lower precision";
  let description = [{
    Quantizes a floating-point tensor to a lower precision tensor.
    Supports parameterized bit_width and quant_scheme (e.g., "ternary").
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
