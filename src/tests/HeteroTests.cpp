#include "HeteroTests.h"
#include "runtime/HeteroPipeline.h"
#include "runtime/net/TenzoComm.h"
#include "runtime/TenzoEngine.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

namespace tenzo {

void runHeteroPipelineTests() {
    llvm::outs() << "=== [Heterogeneous Pipeline] Unit Tests ===\n";

    HeteroPipeline pipeline;

    // Test 1: Valid CPU-only partition
    {
        bool ok = pipeline.parsePartitionString("cpu:0-15", 16);
        assert(ok && "Failed to parse cpu:0-15");
        assert(pipeline.getStages().size() == 1);
        assert(pipeline.getStages()[0].type == HeteroDeviceType::CPU);
        assert(pipeline.getStages()[0].startLayer == 0 && pipeline.getStages()[0].endLayer == 16);
        llvm::outs() << "  [PASS] Single CPU partition (0-15)\n";
    }

    // Test 2: Valid GPU + CPU split
    {
        bool ok = pipeline.parsePartitionString("gpu:0-7,cpu:8-15", 16);
        assert(ok && "Failed to parse gpu:0-7,cpu:8-15");
        assert(pipeline.getStages().size() == 2);
        assert(pipeline.getStages()[0].type == HeteroDeviceType::GPU_VULKAN);
        assert(pipeline.getStages()[0].startLayer == 0 && pipeline.getStages()[0].endLayer == 8);
        assert(pipeline.getStages()[1].type == HeteroDeviceType::CPU);
        assert(pipeline.getStages()[1].startLayer == 8 && pipeline.getStages()[1].endLayer == 16);
        llvm::outs() << "  [PASS] Hybrid GPU (0-7) + CPU (8-15)\n";
    }

    // Test 3: Multi-device including Remote node
    {
        bool ok = pipeline.parsePartitionString("vulkan:0-3,cpu:4-7,remote:127.0.0.1:9000:8-15", 16);
        assert(ok && "Failed to parse vulkan:0-3,cpu:4-7,remote:127.0.0.1:9000:8-15");
        assert(pipeline.getStages().size() == 3);
        assert(pipeline.getStages()[0].type == HeteroDeviceType::GPU_VULKAN);
        assert(pipeline.getStages()[1].type == HeteroDeviceType::CPU);
        assert(pipeline.getStages()[2].type == HeteroDeviceType::REMOTE_NODE);
        assert(pipeline.getStages()[2].remoteHost == "127.0.0.1" && pipeline.getStages()[2].remotePort == 9000);
        assert(pipeline.getStages()[2].startLayer == 8 && pipeline.getStages()[2].endLayer == 16);
        llvm::outs() << "  [PASS] Multi-stage Vulkan (0-3) + CPU (4-7) + Remote (8-15)\n";
    }

    // Test 4: Invalid partition string (out of range layer)
    {
        bool ok = pipeline.parsePartitionString("cpu:0-20", 16);
        assert(!ok && "Should reject layer index > total_layers");
        llvm::outs() << "  [PASS] Out-of-bounds partition correctly rejected\n";
    }

    // Test 5: Auto-partition heuristic
    {
        pipeline.autoPartition(30, /*has_vulkan=*/true, /*has_cuda=*/false, /*has_rocm=*/false, "192.168.1.50:9000");
        assert(pipeline.getStages().size() == 3);
        assert(pipeline.getStages()[0].type == HeteroDeviceType::GPU_VULKAN);
        assert(pipeline.getStages()[1].type == HeteroDeviceType::CPU);
        assert(pipeline.getStages()[2].type == HeteroDeviceType::REMOTE_NODE);
        llvm::outs() << "  [PASS] Auto-partitioning with GPU + CPU + Remote node\n";
        pipeline.printSummary();
    }

    llvm::outs() << "✅ All Heterogeneous Pipeline Unit Tests Passed!\n\n";
}

void runDistributedNetworkTests() {
    llvm::outs() << "=== [Distributed Cluster Protocol] Network Tests ===\n";

    const int testPort = 19532;
    const size_t hiddenDim = 512;

    // Start a TenzoServer on loopback port
    auto server = std::make_unique<net::TenzoServer>(testPort, [](uint16_t startLayer, uint16_t endLayer, const float* inAct, size_t dim, float* outAct) -> bool {
        // Mock layer computation: y = x * 2.0f + (endLayer - startLayer)
        float factor = static_cast<float>(endLayer - startLayer);
        for (size_t i = 0; i < dim; ++i) {
            outAct[i] = inAct[i] * 2.0f + factor;
        }
        return true;
    });

    bool started = server->start();
    assert(started && "Failed to start TenzoServer on loopback port");
    llvm::outs() << "  [PASS] TenzoServer started on 127.0.0.1:" << testPort << "\n";

    // Give server thread a moment to bind and listen
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Connect TenzoClient
    auto client = std::make_unique<net::TenzoClient>("127.0.0.1", testPort);
    bool connected = client->connectToServer();
    assert(connected && "TenzoClient failed to connect to TenzoServer");
    llvm::outs() << "  [PASS] TenzoClient connected successfully\n";

    // Test Ping / RTT
    double rtt = client->ping();
    assert(rtt >= 0.0 && "Ping failed");
    llvm::outs() << "  [PASS] Ping / Latency check: " << rtt << " ms\n";

    // Test Hidden State Activation Transfer and Remote Layer Execution
    std::vector<float> inputActivations(hiddenDim);
    for (size_t i = 0; i < hiddenDim; ++i) {
        inputActivations[i] = static_cast<float>(i + 1) * 0.01f;
    }

    std::vector<float> outputActivations(hiddenDim, 0.0f);
    uint16_t startL = 4, endL = 12; // 8 layers
    bool execOk = client->executeRemoteLayers(1001, startL, endL, inputActivations.data(), hiddenDim, outputActivations.data());
    assert(execOk && "Remote layer execution failed");

    // Verify mathematical correctness of remote result
    float expectedFactor = static_cast<float>(endL - startL);
    float maxDiff = 0.0f;
    for (size_t i = 0; i < hiddenDim; ++i) {
        float expected = inputActivations[i] * 2.0f + expectedFactor;
        float diff = std::abs(outputActivations[i] - expected);
        if (diff > maxDiff) maxDiff = diff;
    }
    assert(maxDiff < 1e-5f && "Remote computation result mismatch");
    llvm::outs() << "  [PASS] Activation transfer (" << (hiddenDim * sizeof(float)) << " bytes) & execution verified (max diff: " << maxDiff << ")\n";

    client->disconnect();
    server->stop();
    llvm::outs() << "  [PASS] Clean disconnect and server shutdown\n";

    llvm::outs() << "✅ All Distributed Cluster Network Tests Passed!\n\n";
}

void runHeteroBenchmark() {
    llvm::outs() << "\n=== [Tenzo Engine] Heterogeneous & Distributed Pipeline Benchmark ===\n";

    // Setup synthetic 30-layer BitNet model configuration
    tenzo_config_t config = tenzo_default_config();
    config.num_layers = 30;
    config.hidden_size = 2048;
    config.ffn_dim = 5632;
    config.num_q_heads = 16;
    config.num_kv_heads = 4;
    config.head_dim = 128;
    config.vocab_size = 32000;

    TenzoEngineImpl engine(config);

    // Profile individual layers and print topology
    engine.hetero_pipeline.parsePartitionString("cpu:0-29", 30);
    llvm::outs() << "\nInitial Baseline (Pure CPU):\n";
    engine.hetero_pipeline.printSummary();

    // Auto-partition across Vulkan GPU + CPU + Remote Node
    engine.hetero_pipeline.autoPartition(30, /*has_vulkan=*/true, /*has_cuda=*/false, /*has_rocm=*/false, "127.0.0.1:9500");
    llvm::outs() << "\nOptimal Heterogeneous Schedule:\n";
    engine.hetero_pipeline.printSummary();

    llvm::outs() << "=== Heterogeneous Benchmark Complete ===\n\n";
}

} // namespace tenzo
