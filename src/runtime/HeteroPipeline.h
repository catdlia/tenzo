#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <iostream>
#include "net/TenzoComm.h"

namespace tenzo {

enum class HeteroDeviceType {
    CPU,
    GPU_VULKAN,
    GPU_CUDA,
    GPU_ROCM,
    REMOTE_NODE
};

struct PipelineStage {
    HeteroDeviceType type = HeteroDeviceType::CPU;
    int deviceIndex = 0;
    int startLayer = 0;
    int endLayer = 0;       // [startLayer, endLayer)
    std::string remoteHost = "127.0.0.1";
    int remotePort = 9200;

    std::shared_ptr<net::TenzoClient> remoteClient;
};

class HeteroPipeline {
public:
    HeteroPipeline();
    ~HeteroPipeline();

    /// Add a stage to the pipeline
    void addStage(const PipelineStage& stage);

    /// Clear all stages
    void clearStages();

    /// Get current configured stages
    const std::vector<PipelineStage>& getStages() const { return stages; }

    /// Helper to parse partition string like "gpu:0:0-14,cpu:15-29" or "vulkan:0-10,remote:192.168.1.50:9200:11-29"
    bool parsePartitionString(const std::string& spec, int totalLayers);

    /// Auto-partition layers across available hardware according to profiled bandwidth
    void autoPartition(int totalLayers, bool hasVulkan, bool hasCuda, bool hasRocm, const std::string& remoteNode = "");

    /// Pretty print current partition configuration
    void printTopology() const;
    void printSummary() const { printTopology(); }

private:
    std::vector<PipelineStage> stages;
};

} // namespace tenzo
