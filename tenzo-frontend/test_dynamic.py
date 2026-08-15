import onnx
import numpy as np
from onnx import helper, TensorProto
import sys
import os

# Ensure tenzo-frontend is in path
sys.path.append(os.path.join(os.getcwd(), 'tenzo-frontend'))
from onnx_to_mlir import onnx_to_mlir

def create_dynamic_model(output_path):
    # Dynamic batch size: [None, 128] * [128, 64] -> [None, 64]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', 128])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', 64])
    
    w = np.random.randn(128, 64).astype(np.float32)
    w_init = helper.make_tensor('W', TensorProto.FLOAT, [128, 64], w.flatten())
    
    node = helper.make_node('MatMul', ['input', 'W'], ['output'])
    
    graph = helper.make_graph(
        [node],
        'dynamic_matmul',
        [input_tensor],
        [output_tensor],
        [w_init]
    )
    
    model = helper.make_model(graph, producer_name='tenzo-test')
    onnx.save(model, output_path)
    print(f"Created dynamic ONNX at {output_path}")

if __name__ == "__main__":
    onnx_path = 'dynamic_test.onnx'
    mlir_path = 'dynamic_test.mlir'
    
    create_dynamic_model(onnx_path)
    onnx_to_mlir(onnx_path, mlir_path)
    
    if os.path.exists(mlir_path):
        print(f"\n✅ Success: {mlir_path} generated.")
        with open(mlir_path, 'r') as f:
            print("--- MLIR Snippet ---")
            print(f.read())
    else:
        print(f"\n❌ Failed: {mlir_path} not generated.")
