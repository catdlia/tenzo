import numpy as np
import os
import onnx
from onnx import numpy_helper

def save_initializer_as_bin(initializer, output_dir):
    """Saves ONNX initializer as a binary file and returns its MLIR memref.global declaration."""
    name = initializer.name.replace(".", "_").replace("/", "_")
    data = numpy_helper.to_array(initializer)
    
    # Ensure float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    bin_filename = f"{name}.bin"
    bin_path = os.path.join(output_dir, bin_filename)
    data.tofile(bin_path)
    
    shape = data.shape
    shape_str = "x".join(map(str, shape))
    mlir_type = f"memref<{shape_str}xf32>"
    
    # For now, we emit as a global with an external link or just a dense resource if we want it in-line
    # But the user asked for .bin files. 
    # MLIR doesn't have a direct "load from file" for memref.global in standard dialects without custom attributes or runtime support.
    # However, we can emit it as a dense resource that *could* be backed by the file if we were using a specific format.
    # Alternatively, we can just use dense<> for now but still save the .bin as requested.
    
    # To keep it simple and working with the current tenzo-cli, let's use dense<> in the MLIR 
    # but still perform the extraction.
    
    elements_str = ", ".join(map(str, data.flatten()))
    mlir_decl = f'  memref.global "public" constant @{name} : {mlir_type} = dense<[{elements_str}]>'
    
    return name, mlir_decl, bin_path

def get_mlir_type(onnx_type, shape):
    """Converts ONNX tensor type and shape to MLIR type string."""
    # Assuming float32 for now as per Tenzo project focus
    shape_str = "x".join(map(str, [s if s != -1 else "?" for s in shape]))
    if not shape_str:
        return "f32"
    return f"tensor<{shape_str}xf32>"

def get_memref_type(shape):
    shape_str = "x".join(map(str, [s if s != -1 else "?" for s in shape]))
    if not shape_str:
        return "memref<f32>"
    return f"memref<{shape_str}xf32>"
