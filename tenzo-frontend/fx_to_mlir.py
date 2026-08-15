import os
import sys
import operator

libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".python_libs"))
if os.path.exists(libs_dir) and libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)



try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Ensure PyTorch is optionally importable or handled safely
try:
    import torch
    import torch.nn as nn
    import torch.fx as fx
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from qat import BitLinear, TernaryQuantizeSTE
except ImportError:
    BitLinear = None
    TernaryQuantizeSTE = None


def pack_ternary_list(flat_list) -> bytes:
    """
    Packs a python list of ternary values {-1, 0, 1} into 2-bit representation.
    4 values per byte (uint8).
    
    Encoding:
      0  -> 0b00 (0)
      1  -> 0b01 (1)
     -1  -> 0b10 (2)
    """
    # Pad to multiple of 4
    remainder = len(flat_list) % 4
    if remainder != 0:
        flat_list = flat_list + [0] * (4 - remainder)

    packed_bytes = bytearray()
    for i in range(0, len(flat_list), 4):
        vals = flat_list[i:i+4]
        b = 0
        for shift_idx, v in enumerate(vals):
            enc = 0
            if v == 1:
                enc = 1
            elif v == -1:
                enc = 2
            b |= (enc & 0x03) << (shift_idx * 2)
        packed_bytes.append(b)
        
    return bytes(packed_bytes)


def pack_ternary_array(arr) -> bytes:
    if HAS_NUMPY and isinstance(arr, np.ndarray):
        flat_list = arr.flatten().tolist()
    else:
        flat_list = list(arr)
    return pack_ternary_list(flat_list)



class FXToMLIREmitter:
    def __init__(self, model: "torch.nn.Module", output_dir: str = "."):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for FXToMLIREmitter")

        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.weights_bin_path = os.path.join(self.output_dir, "weights.bin")
        self.weights_file = open(self.weights_bin_path, "wb")
        self.current_byte_offset = 0

        self.weight_metadata = {}  # name -> {offset, size_bytes, shape, orig_dtype}
        self.mlir_lines = []
        self.ssa_map = {}
        self.type_map = {}
        self.var_counter = 0

    def _get_new_var(self, mlir_type: str = "tensor<*xf32>") -> str:
        var_name = f"%v{self.var_counter}"
        self.var_counter += 1
        self.type_map[var_name] = mlir_type
        return var_name

    def _get_dim(self, t_type: str, dim_idx: int) -> int:
        dims = t_type.replace("tensor<", "").replace("xf32>", "").split("x")
        dims = [d for d in dims if d and d != "?"]
        if not dims: return 0
        return int(dims[dim_idx])

    def _save_weight(self, name: str, tensor: "torch.Tensor", is_ternary: bool = False):
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        shape = list(arr.shape)

        if is_ternary:
            scale = float(np.abs(arr).mean())
            if scale < 1e-5:
                scale = 1.0
            quant_arr = np.clip(np.round(arr / scale), -1, 1).astype(np.int8)
            packed_data = pack_ternary_array(quant_arr)
            quant_scheme = "ternary"
            bit_width = 2
            data_to_store = quant_arr
        else:
            scale = 1.0
            quant_scheme = "none"
            bit_width = 32
            packed_data = arr.tobytes()
            data_to_store = arr

        offset = self.current_byte_offset
        size_bytes = len(packed_data)
        self.weights_file.write(packed_data)
        self.current_byte_offset += size_bytes

        meta = {
            "name": name,
            "offset": offset,
            "size_bytes": size_bytes,
            "shape": shape,
            "scale": scale,
            "bit_width": bit_width,
            "quant_scheme": quant_scheme,
            "_data": data_to_store,
        }
        self.weight_metadata[name] = meta
        return meta

    def _make_dense_attr(self, data: "np.ndarray", shape: list) -> str:
        """Generate MLIR dense<[...]> attribute from numpy float32 array."""
        flat = data.flatten().tolist()
        vals = ", ".join(f"{v:.8e}" for v in flat)
        if len(shape) == 1:
            return f"dense<[{vals}]>"
        # For multi-dim, wrap in nested lists matching shape
        def nest(lst, dims):
            if len(dims) == 1:
                return "[" + ", ".join(f"{x:.8e}" for x in lst) + "]"
            stride = 1
            for d in dims[1:]:
                stride *= d
            chunks = [lst[i*stride:(i+1)*stride] for i in range(dims[0])]
            return "[" + ", ".join(nest(c, dims[1:]) for c in chunks) + "]"
        return f"dense<{nest(flat, shape)}>"

    def convert(self, sample_input: "torch.Tensor" = None) -> str:
        class TenzoTracer(fx.Tracer):
            def is_leaf_module(self, m: "torch.nn.Module", module_qualified_name: str) -> bool:
                if BitLinear is not None and isinstance(m, BitLinear):
                    return True
                if isinstance(m, (nn.Embedding, nn.LayerNorm)):
                    return True
                if m.__class__.__name__ in ("RotaryEmbedding", "BitLinear", "LlamaRMSNorm", "RMSNorm", "Embedding"):
                    return True
                return super().is_leaf_module(m, module_qualified_name)

        tracer = TenzoTracer()
        graph = tracer.trace(self.model)

        self.mlir_lines.append("// Generated by Tenzo PyTorch FX Frontend (Zero-Copy Bridge)")
        self.mlir_lines.append("module {")

        sample_rank = len(sample_input.shape) if sample_input is not None else 2

        # 1. Inspect parameters & submodules
        for name, param in self.model.named_parameters():
            is_bitlinear = False
            # Check if this weight belongs to a BitLinear module
            parent_module_name = name.rsplit(".", 1)[0] if "." in name else ""
            if parent_module_name:
                mod = dict(self.model.named_modules()).get(parent_module_name)
                if isinstance(mod, BitLinear):
                    is_bitlinear = True

            self._save_weight(name, param, is_ternary=is_bitlinear)

        self.mlir_lines.append("")

        # 2. Main function creation
        input_args = []
        input_types = []
        arg_idx = 0

        for node in graph.nodes:
            if node.op == "placeholder":
                shape_str = "1x10"  # Default fallback if sample input shape not known
                if sample_input is not None:
                    shape_str = "x".join(map(str, list(sample_input.shape)))
                
                if sample_input is not None and sample_input.dtype in (torch.int32, torch.int64):
                    t_type = f"tensor<{shape_str}xi32>"
                else:
                    t_type = f"tensor<{shape_str}xf32>"

                ssa_name = f"%arg{arg_idx}"
                arg_idx += 1
                input_args.append(f"{ssa_name}: {t_type}")
                input_types.append(t_type)
                self.ssa_map[node] = ssa_name
                self.type_map[ssa_name] = t_type

        # Add mmap weights buffer argument
        weights_buf_arg = f"%arg{arg_idx}"
        arg_idx += 1
        input_args.append(f"{weights_buf_arg}: memref<?xi8>")
        self.type_map[weights_buf_arg] = "memref<?xi8>"

        # Add KV cache arguments
        embed_dim = 128
        if sample_input is not None:
            embed_dim = sample_input.shape[-1]
            
        num_heads = None
        head_dim = None
        for m in self.model.modules():
            if hasattr(m, "num_heads") and hasattr(m, "head_dim"):
                num_heads = m.num_heads
                head_dim = m.head_dim
                break

        if num_heads is not None and head_dim is not None:
            cache_k_type = f"tensor<1x{num_heads}x1024x{head_dim}xf32>"
            cache_v_type = f"tensor<1x{num_heads}x1024x{head_dim}xf32>"
        else:
            cache_k_type = f"tensor<1x1024x{embed_dim}xf32>"
            cache_v_type = f"tensor<1x1024x{embed_dim}xf32>"

        seq_pos_type = f"tensor<1xi32>"
        
        cache_k_arg = f"%arg{arg_idx}"
        arg_idx += 1
        cache_v_arg = f"%arg{arg_idx}"
        arg_idx += 1
        seq_pos_arg = f"%arg{arg_idx}"
        arg_idx += 1
        
        input_args.append(f"{cache_k_arg}: {cache_k_type}")
        input_args.append(f"{cache_v_arg}: {cache_v_type}")
        input_args.append(f"{seq_pos_arg}: {seq_pos_type}")
        
        self.type_map[cache_k_arg] = cache_k_type
        self.type_map[cache_v_arg] = cache_v_type
        self.type_map[seq_pos_arg] = seq_pos_type

        # Output signature placeholder (Input/Output tensor + updated cache K + updated cache V)
        func_sig = f"  func.func @main({', '.join(input_args)}) -> (__OUT_TYPE__, {cache_k_type}, {cache_v_type}) attributes {{llvm.emit_c_interface}} {{"
        self.mlir_lines.append(func_sig)

        current_cache_k = cache_k_arg
        current_cache_v = cache_v_arg

        # 3. Process FX Graph nodes
        for node in graph.nodes:
            if node.op in ("placeholder", "get_attr"):
                continue

            elif node.op == "call_module":
                submod = dict(self.model.named_modules()).get(node.target)
                if submod.__class__.__name__ == "RotaryEmbedding":
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    res_var = self._get_new_var(inp_type)
                    self.mlir_lines.append(
                        f'    {res_var} = "tenzo.rope"({inp}, {seq_pos_arg}) : ({inp_type}, {seq_pos_type}) -> {inp_type}'
                    )
                    self.ssa_map[node] = res_var
                    self.type_map[res_var] = inp_type

                elif isinstance(submod, nn.Embedding) or submod.__class__.__name__ == "Embedding":
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    
                    w_name = f"{node.target}.weight"
                    w_meta = self.weight_metadata[w_name]
                    w_offset = w_meta["offset"]
                    w_shape = w_meta["shape"]
                    
                    w_shape_str = f"{w_shape[0]}x{w_shape[1]}"
                    c_off_var = self._get_new_var("index")
                    self.mlir_lines.append(f'    {c_off_var} = arith.constant {w_offset} : index')
                    
                    w_memref_var = self._get_new_var(f"memref<{w_shape_str}xf32>")
                    self.mlir_lines.append(f'    {w_memref_var} = memref.view {weights_buf_arg}[{c_off_var}][] : memref<?xi8> to memref<{w_shape_str}xf32>')
                    w_tensor_var = self._get_new_var(f"tensor<{w_shape_str}xf32>")
                    self.mlir_lines.append(f'    {w_tensor_var} = bufferization.to_tensor {w_memref_var} : memref<{w_shape_str}xf32> to tensor<{w_shape_str}xf32>')
                    
                    res_type = f"tensor<1x1x{w_shape[1]}xf32>"
                    res_var = self._get_new_var(res_type)
                    self.mlir_lines.append(
                        f'    {res_var} = "tenzo.embedding"({inp}, {w_tensor_var}) : ({inp_type}, tensor<{w_shape_str}xf32>) -> {res_type}'
                    )
                    self.ssa_map[node] = res_var
                    self.type_map[res_var] = res_type

                elif submod.__class__.__name__ in ("LlamaRMSNorm", "RMSNorm"):
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    
                    w_name = f"{node.target}.weight"
                    w_meta = self.weight_metadata[w_name]
                    w_offset = w_meta["offset"]
                    w_shape = w_meta["shape"]
                    w_shape_str = "x".join(map(str, w_shape))
                    
                    c_off_var = self._get_new_var("index")
                    self.mlir_lines.append(f'    {c_off_var} = arith.constant {w_offset} : index')
                    
                    w_memref_var = self._get_new_var(f"memref<{w_shape_str}xf32>")
                    self.mlir_lines.append(f'    {w_memref_var} = memref.view {weights_buf_arg}[{c_off_var}][] : memref<?xi8> to memref<{w_shape_str}xf32>')
                    w_tensor_var = self._get_new_var(f"tensor<{w_shape_str}xf32>")
                    self.mlir_lines.append(f'    {w_tensor_var} = bufferization.to_tensor {w_memref_var} : memref<{w_shape_str}xf32> to tensor<{w_shape_str}xf32>')
                    
                    eps_val = getattr(submod, "variance_epsilon", 1e-6)
                    res_var = self._get_new_var(inp_type)
                    self.mlir_lines.append(
                        f'    {res_var} = "tenzo.rmsnorm"({inp}, {w_tensor_var}) {{eps = {eps_val:.8e} : f32}} : ({inp_type}, tensor<{w_shape_str}xf32>) -> {inp_type}'
                    )
                    self.ssa_map[node] = res_var
                    self.type_map[res_var] = inp_type

                elif isinstance(submod, BitLinear):
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    
                    w_name = f"{node.target}.weight"
                    w_meta = self.weight_metadata[w_name]
                    w_offset = w_meta["offset"]
                    w_shape = w_meta["shape"]
                    w_scale = w_meta["scale"]
                    
                    # Packed 2-bit shape: [out_features, in_features // 4]
                    packed_shape = [w_shape[0], w_shape[1] // 4]
                    packed_shape_str = f"{packed_shape[0]}x{packed_shape[1]}"
                    
                    c_off_var = self._get_new_var("index")
                    self.mlir_lines.append(f'    {c_off_var} = arith.constant {w_offset} : index')
                    
                    w_memref_type = f"memref<{packed_shape_str}xi8>"
                    w_tensor_type = f"tensor<{packed_shape_str}xi8>"
                    
                    w_memref_var = self._get_new_var(w_memref_type)
                    self.mlir_lines.append(f'    {w_memref_var} = memref.view {weights_buf_arg}[{c_off_var}][] : memref<?xi8> to {w_memref_type}')
                    
                    w_tensor_var = self._get_new_var(w_tensor_type)
                    self.mlir_lines.append(f'    {w_tensor_var} = bufferization.to_tensor {w_memref_var} : {w_memref_type} to {w_tensor_type}')
                    
                    # Scale constant
                    scale_var = self._get_new_var("f32")
                    self.mlir_lines.append(f'    {scale_var} = arith.constant {w_scale:.8e} : f32')
                    
                    # Dequantize Op
                    fp32_w_type = f"tensor<{w_shape[0]}x{w_shape[1]}xf32>"
                    q_var = self._get_new_var(fp32_w_type)
                    self.mlir_lines.append(
                        f'    {q_var} = "tenzo.dequantize"({w_tensor_var}, {scale_var}) {{bit_width = 2 : i32, quant_scheme = "ternary"}} : '
                        f'({w_tensor_type}, f32) -> {fp32_w_type}'
                    )

                    inp_dims = [s for s in inp_type.replace("tensor<", "").replace("xf32>", "").split("x") if s]
                    out_shape_dims = inp_dims[:-1] + [str(w_shape[0])]
                    out_shape_str = "x".join(out_shape_dims)

                    mm_type = f"tensor<{out_shape_str}xf32>"
                    mm_var = self._get_new_var(mm_type)

                    self.mlir_lines.append(
                        f'    {mm_var} = "tenzo.matmul"({inp}, {q_var}) : ({inp_type}, {fp32_w_type}) -> {mm_type}'
                    )

                    if submod.bias is not None:
                        b_name = f"{node.target}.bias"
                        b_meta = self.weight_metadata[b_name]
                        b_offset = b_meta["offset"]
                        b_shape = b_meta["shape"]
                        p_shape = list(b_shape)
                        if len(p_shape) == 1 and len(inp_dims) > 1:
                            p_shape = [1] * (len(inp_dims) - 1) + p_shape

                        b_shape_str = "x".join(map(str, p_shape))
                        b_memref_type = f"memref<{b_shape_str}xf32>"
                        b_tensor_type = f"tensor<{b_shape_str}xf32>"
                        
                        b_off_var = self._get_new_var("index")
                        self.mlir_lines.append(f'    {b_off_var} = arith.constant {b_offset} : index')
                        
                        b_memref_var = self._get_new_var(b_memref_type)
                        self.mlir_lines.append(f'    {b_memref_var} = memref.view {weights_buf_arg}[{b_off_var}][] : memref<?xi8> to {b_memref_type}')
                        b_tensor_var = self._get_new_var(b_tensor_type)
                        self.mlir_lines.append(f'    {b_tensor_var} = bufferization.to_tensor {b_memref_var} : {b_memref_type} to {b_tensor_type}')
                        
                        add_var = self._get_new_var(mm_type)
                        self.mlir_lines.append(
                            f'    {add_var} = "tenzo.add"({mm_var}, {b_tensor_var}) : ({mm_type}, {b_tensor_type}) -> {mm_type}'
                        )
                        self.ssa_map[node] = add_var
                    else:
                        self.ssa_map[node] = mm_var

                elif isinstance(submod, nn.ReLU):
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    res_var = self._get_new_var(inp_type)
                    self.mlir_lines.append(f'    {res_var} = "tenzo.relu"({inp}) : ({inp_type}) -> {inp_type}')
                    self.ssa_map[node] = res_var

            elif node.op == "call_method":
                method_name = node.target
                if method_name in ("view", "reshape"):
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    inp_shape_dims = [int(s) for s in inp_type.replace("tensor<", "").replace("xf32>", "").split("x")]

                    raw_args = []
                    for arg in node.args[1:]:
                        if isinstance(arg, (tuple, list)):
                            raw_args.extend(arg)
                        else:
                            raw_args.append(arg)

                    target_shape = []
                    for arg in raw_args:
                        if isinstance(arg, int):
                            target_shape.append(arg)
                        elif hasattr(arg, "target") and isinstance(arg.target, int):
                            target_shape.append(arg.target)
                        elif isinstance(arg, torch.fx.Node) and hasattr(arg, "meta") and "val" in arg.meta:
                            try:
                                target_shape.append(int(arg.meta["val"]))
                            except Exception:
                                target_shape.append(-1)
                        else:
                            target_shape.append(-1)

                    total_elems = 1
                    for d in inp_shape_dims:
                        total_elems *= d

                    num_unknown = sum(1 for d in target_shape if d <= 0)
                    if num_unknown == 1:
                        idx = [i for i, d in enumerate(target_shape) if d <= 0][0]
                        known_prod = 1
                        for i, d in enumerate(target_shape):
                            if i != idx and d > 0:
                                known_prod *= d
                        target_shape[idx] = total_elems // known_prod
                    elif num_unknown > 1:
                        if len(target_shape) == 3:
                            target_shape = [1, 1, total_elems]
                        elif len(target_shape) == 4:
                            target_shape = [1, 1, 4, total_elems // 4]

                    target_shape_str = "x".join(map(str, target_shape))
                    out_type = f"tensor<{target_shape_str}xf32>"
                    res_var = self._get_new_var(out_type)

                    if len(inp_shape_dims) == 3 and len(target_shape) == 4:
                        self.mlir_lines.append(
                            f'    {res_var} = tensor.expand_shape {inp} [[0], [1], [2, 3]] output_shape [{", ".join(map(str, target_shape))}] : {inp_type} into {out_type}'
                        )
                    elif len(inp_shape_dims) == 4 and len(target_shape) == 3:
                        self.mlir_lines.append(
                            f'    {res_var} = tensor.collapse_shape {inp} [[0], [1], [2, 3]] : {inp_type} into {out_type}'
                        )
                    else:
                        res_var = inp
                        out_type = inp_type

                    self.ssa_map[node] = res_var
                    self.type_map[res_var] = out_type

                elif method_name == "transpose":
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    inp_shape_dims = [int(s) for s in inp_type.replace("tensor<", "").replace("xf32>", "").split("x")]
                    dim1 = node.args[1]
                    dim2 = node.args[2]
                    if dim1 < 0: dim1 += len(inp_shape_dims)
                    if dim2 < 0: dim2 += len(inp_shape_dims)

                    perm = list(range(len(inp_shape_dims)))
                    perm[dim1], perm[dim2] = perm[dim2], perm[dim1]

                    out_shape_dims = [inp_shape_dims[p] for p in perm]
                    out_shape_str = "x".join(map(str, out_shape_dims))
                    out_type = f"tensor<{out_shape_str}xf32>"

                    empty_var = self._get_new_var(out_type)
                    self.mlir_lines.append(f'    {empty_var} = tensor.empty() : {out_type}')

                    res_var = self._get_new_var(out_type)
                    perm_str = ", ".join(map(str, perm))
                    self.mlir_lines.append(
                        f'    {res_var} = linalg.transpose ins({inp} : {inp_type}) outs({empty_var} : {out_type}) permutation = [{perm_str}]'
                    )
                    self.ssa_map[node] = res_var
                    self.type_map[res_var] = out_type

                elif method_name in ("contiguous", "to"):
                    inp = self.ssa_map[node.args[0]]
                    self.ssa_map[node] = inp
                    self.type_map[inp] = self.type_map[inp]


            elif node.op == "call_function":
                target = node.target
                if target in (torch.relu, F.relu):
                    inp = self.ssa_map[node.args[0]]
                    inp_type = self.type_map[inp]
                    res_var = self._get_new_var(inp_type)
                    self.mlir_lines.append(f'    {res_var} = "tenzo.relu"({inp}) : ({inp_type}) -> {inp_type}')
                    self.ssa_map[node] = res_var

                elif target in (torch.add, operator.add, getattr(torch, "add", None)):

                    lhs = self.ssa_map[node.args[0]]
                    rhs = self.ssa_map[node.args[1]]
                    inp_type = self.type_map[lhs]
                    res_var = self._get_new_var(inp_type)
                    self.mlir_lines.append(f'    {res_var} = "tenzo.add"({lhs}, {rhs}) : ({inp_type}, {inp_type}) -> {inp_type}')
                    self.ssa_map[node] = res_var

                elif target in (F.scaled_dot_product_attention, getattr(torch._C._nn, "scaled_dot_product_attention", None)):
                    q = self.ssa_map[node.args[0]]
                    k = self.ssa_map[node.args[1]]
                    v = self.ssa_map[node.args[2]]
                    q_type = self.type_map[q]
                    k_type = self.type_map[k]
                    v_type = self.type_map[v]
                    
                    # 1. Update KV Cache
                    updated_k = self._get_new_var(cache_k_type)
                    updated_v = self._get_new_var(cache_v_type)
                    
                    self.mlir_lines.append(
                        f'    {updated_k}, {updated_v} = "tenzo.kv_cache_update"({current_cache_k}, {current_cache_v}, {k}, {v}, {seq_pos_arg}) : '
                        f'({cache_k_type}, {cache_v_type}, {k_type}, {v_type}, {seq_pos_type}) -> ({cache_k_type}, {cache_v_type})'
                    )
                    
                    current_cache_k = updated_k
                    current_cache_v = updated_v

                    # 2. Extract valid cache slice to pass to attention
                    # cache shape: [1, 1024, 128], q shape: [1, seq_len, 128]
                    # We need to extract [1, seq_pos + current_seq_len, 128]
                    # But for now, to keep the compiler lowering simple, we will just pass the updated cache and the seq_pos to a new Tenzo Attention op, OR we can extract slice.
                    # Since we don't have tensor.extract_slice lowering fully wired for dynamic sizes easily in frontend without emitting lots of ops, let's just pass seq_pos to tenzo.attention!
                    
                    res_var = self._get_new_var(q_type)
                    self.mlir_lines.append(
                        f'    {res_var} = "tenzo.attention"({q}, {updated_k}, {updated_v}, {seq_pos_arg}) : ({q_type}, {cache_k_type}, {cache_v_type}, {seq_pos_type}) -> {q_type}'
                    )
                    self.ssa_map[node] = res_var


            elif node.op == "output":
                res_val = self.ssa_map[node.args[0]]
                res_type = self.type_map[res_val]
                self.mlir_lines.append(f'    return {res_val}, {current_cache_k}, {current_cache_v} : {res_type}, {cache_k_type}, {cache_v_type}')

        self.mlir_lines.append("  }")
        self.mlir_lines.append("}")

        self.weights_file.close()
        
        mlir_content = "\n".join(self.mlir_lines)
        mlir_content = mlir_content.replace("__OUT_TYPE__", res_type)

        mlir_out_path = os.path.join(self.output_dir, "model.mlir")
        with open(mlir_out_path, "w") as f:
            f.write(mlir_content)

        print(f"[FX-to-MLIR] Successfully exported model to {mlir_out_path}")
        print(f"[FX-to-MLIR] Binary weights saved to {self.weights_bin_path} ({self.current_byte_offset} bytes)")
        return mlir_content


def export_torch_model_to_tenzo(model: "torch.nn.Module", sample_input: "torch.Tensor", output_dir: str = "."):
    emitter = FXToMLIREmitter(model, output_dir=output_dir)
    return emitter.convert(sample_input=sample_input)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "export_output")
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "weights.bin")
    mlir_path = os.path.join(out_dir, "model.mlir")

    if HAS_TORCH:
        from qat import BitLinear

        class BitSelfAttention(nn.Module):
            def __init__(self, embed_dim=128):
                super().__init__()
                self.q_proj = BitLinear(embed_dim, embed_dim)
                self.k_proj = BitLinear(embed_dim, embed_dim)
                self.v_proj = BitLinear(embed_dim, embed_dim)
                self.out_proj = BitLinear(embed_dim, embed_dim)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                out = self.out_proj(attn_out)
                return out

        class TransformerBlock(nn.Module):
            def __init__(self, embed_dim=128, ff_dim=256):
                super().__init__()
                self.attn = BitSelfAttention(embed_dim)
                self.fc1 = BitLinear(embed_dim, ff_dim)
                self.relu = nn.ReLU()
                self.fc2 = BitLinear(ff_dim, embed_dim)

            def forward(self, x):
                h = x + self.attn(x)
                out = h + self.fc2(self.relu(self.fc1(h)))
                return out

        model = TransformerBlock(embed_dim=128, ff_dim=256)
        # Sequence length must be 1 for autoregressive generation
        sample_input = torch.randn(1, 1, 128) # Batch=1, SeqLen=1, Dim=128
        
        # Save sample input and expected output for C++ validation
        with torch.no_grad():
            expected_output = model(sample_input)
            
        input_bin_path = os.path.join(out_dir, "input.bin")
        expected_bin_path = os.path.join(out_dir, "expected.bin")
        
        with open(input_bin_path, "wb") as f:
            f.write(sample_input.numpy().astype(np.float32).tobytes())
            
        with open(expected_bin_path, "wb") as f:
            f.write(expected_output.numpy().astype(np.float32).tobytes())
            
        print(f"[FX-to-MLIR] Saved sample input to {input_bin_path}")
        print(f"[FX-to-MLIR] Saved expected output to {expected_bin_path}")
        
        export_torch_model_to_tenzo(model, sample_input, output_dir=out_dir)

    else:
        print("[FX-to-MLIR] PyTorch not detected. Running Standalone Pure-Python Mock Exporter...")
        
        # 1. Generate packed weights (128->64 -> 10)
        dummy_weights = [1, -1, 0, 1] * 64
        packed_bytes = pack_ternary_list(dummy_weights)
        
        with open(bin_path, "wb") as f:
            f.write(packed_bytes)

        # Generate dummy input and expected output for C++ validation
        import struct
        input_bin_path = os.path.join(out_dir, "input.bin")
        expected_bin_path = os.path.join(out_dir, "expected.bin")
        dummy_input = [0.5] * 10
        dummy_expected = [1.0] * 20
        
        with open(input_bin_path, "wb") as f:
            f.write(struct.pack('10f', *dummy_input))
        with open(expected_bin_path, "wb") as f:
            f.write(struct.pack('20f', *dummy_expected))

        print(f"[FX-to-MLIR] Saved sample input to {input_bin_path}")
        print(f"[FX-to-MLIR] Saved expected output to {expected_bin_path}")

        # 2. Generate model.mlir
        mlir_content = """// Standalone Mock MLIR model for Zero-Copy Bridge Test
module {
  memref.global "public" constant @fc1_weight : memref<20x10xf32> = dense<0.0> {bit_width = 2 : i32, offset = 0 : i64, quant_scheme = "ternary", size_bytes = 64 : i64}

  func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1x20xf32> {
    %0 = memref.get_global @fc1_weight : memref<20x10xf32>
    %1 = bufferization.to_tensor %0 : memref<20x10xf32> to tensor<20x10xf32>
    %2 = "tenzo.quantize"(%1) {bit_width = 2 : i32, quant_scheme = "ternary"} : (tensor<20x10xf32>) -> tensor<20x10xf32>
    %3 = "tenzo.matmul"(%arg0, %2) : (tensor<1x10xf32>, tensor<20x10xf32>) -> tensor<1x20xf32>
    return %3 : tensor<1x20xf32>
  }
}
"""
        with open(mlir_path, "w") as f:
            f.write(mlir_content)

        print(f"[FX-to-MLIR] Successfully generated mock model at {mlir_path}")
        print(f"[FX-to-MLIR] Binary weights saved to {bin_path} ({len(packed_bytes)} bytes)")

