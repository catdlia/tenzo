import onnx
import numpy as np
from onnx import helper, TensorProto
import sys
import os

# Ensure tenzo-frontend is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from onnx_to_mlir import onnx_to_mlir

def create_simple_mlp(output_path):
    # Linear -> ReLU -> Linear
    # Input: [1, 10]
    # W1: [10, 20], B1: [20]
    # W2: [20, 5], B2: [5]
    # Output: [1, 5]
    
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5])
    
    w1 = np.random.randn(10, 20).astype(np.float32)
    b1 = np.random.randn(20).astype(np.float32)
    w2 = np.random.randn(20, 5).astype(np.float32)
    b2 = np.random.randn(5).astype(np.float32)
    
    w1_init = helper.make_tensor('W1', TensorProto.FLOAT, [10, 20], w1.flatten())
    b1_init = helper.make_tensor('B1', TensorProto.FLOAT, [20], b1.flatten())
    w2_init = helper.make_tensor('W2', TensorProto.FLOAT, [20, 5], w2.flatten())
    b2_init = helper.make_tensor('B2', TensorProto.FLOAT, [5], b2.flatten())
    
    # We use Gemm node. In ONNX, Gemm(A, B, C) = alpha*A*B + beta*C
    # Default alpha=1.0, beta=1.0
    node1 = helper.make_node('Gemm', ['input', 'W1', 'B1'], ['t1'])
    node2 = helper.make_node('Relu', ['t1'], ['t2'])
    node3 = helper.make_node('Gemm', ['t2', 'W2', 'B2'], ['output'])
    
    graph = helper.make_graph(
        [node1, node2, node3],
        'simple_mlp',
        [input_tensor],
        [output_tensor],
        [w1_init, b1_init, w2_init, b2_init]
    )
    
    model = helper.make_model(graph, producer_name='tenzo-test')
    onnx.save(model, output_path)
    print(f"Created simple MLP ONNX at {output_path}")

if __name__ == "__main__":
    onnx_path = 'simple_mlp.onnx'
    mlir_path = 'simple_mlp.mlir'
    
    create_simple_mlp(onnx_path)
    onnx_to_mlir(onnx_path, mlir_path)
    
    if os.path.exists(mlir_path):
        print(f"\n✅ Test passed: {mlir_path} generated.")
        print("\n--- MLIR Snippet ---")
        with open(mlir_path, 'r') as f:
            lines = f.readlines()
            for line in lines[:20]: # Show more lines to see nodes
                print(line.strip())
        print("--------------------")
    else:
        print(f"\n❌ Test failed: {mlir_path} not generated.")
        sys.exit(1)
