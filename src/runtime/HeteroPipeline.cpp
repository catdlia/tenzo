#include "HeteroPipeline.h"
#include <sstream>
#include <algorithm>

namespace tenzo {

HeteroPipeline::HeteroPipeline() = default;
HeteroPipeline::~HeteroPipeline() = default;

void HeteroPipeline::addStage(const PipelineStage& stage) {
    stages.push_back(stage);
}

void HeteroPipeline::clearStages() {
    stages.clear();
}

bool HeteroPipeline::parsePartitionString(const std::string& spec, int totalLayers) {
    if (spec.empty() || totalLayers <= 0) return false;

    clearStages();
    std::stringstream ss(spec);
    std::string token;

    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;

        // Examples:
        // "cpu:0-29"
        // "gpu:0:0-14"
        // "vulkan:0-14"
        // "cuda:0-14"
        // "rocm:0-14"
        // "remote:192.168.1.50:9200:15-29"

        std::vector<std::string> parts;
        std::stringstream token_ss(token);
        std::string sub;
        while (std::getline(token_ss, sub, ':')) {
            parts.push_back(sub);
        }

        if (parts.size() < 2) continue;

        PipelineStage stage;
        std::string devName = parts[0];

        if (devName == "cpu") {
            stage.type = HeteroDeviceType::CPU;
            // cpu:0-29
            std::string range = parts[1];
            size_t dash = range.find('-');
            if (dash != std::string::npos) {
                stage.startLayer = std::stoi(range.substr(0, dash));
                stage.endLayer = std::stoi(range.substr(dash + 1)) + 1;
            }
        } else if (devName == "gpu" || devName == "vulkan") {
            stage.type = HeteroDeviceType::GPU_VULKAN;
            if (parts.size() == 2) {
                // vulkan:0-14
                std::string range = parts[1];
                size_t dash = range.find('-');
                if (dash != std::string::npos) {
                    stage.startLayer = std::stoi(range.substr(0, dash));
                    stage.endLayer = std::stoi(range.substr(dash + 1)) + 1;
                }
            } else if (parts.size() >= 3) {
                // gpu:0:0-14
                stage.deviceIndex = std::stoi(parts[1]);
                std::string range = parts[2];
                size_t dash = range.find('-');
                if (dash != std::string::npos) {
                    stage.startLayer = std::stoi(range.substr(0, dash));
                    stage.endLayer = std::stoi(range.substr(dash + 1)) + 1;
                }
            }
        } else if (devName == "cuda") {
            stage.type = HeteroDeviceType::GPU_CUDA;
            std::string range = parts[1];
            size_t dash = range.find('-');
            if (dash != std::string::npos) {
                stage.startLayer = std::stoi(range.substr(0, dash));
                stage.endLayer = std::stoi(range.substr(dash + 1)) + 1;
            }
        } else if (devName == "rocm") {
            stage.type = HeteroDeviceType::GPU_ROCM;
            std::string range = parts[1];
            size_t dash = range.find('-');
            if (dash != std::string::npos) {
                stage.startLayer = std::stoi(range.substr(0, dash));
                stage.endLayer = std::stoi(range.substr(dash + 1)) + 1;
            }
        } else if (devName == "remote") {
            stage.type = HeteroDeviceType::REMOTE_NODE;
            if (parts.size() >= 4) {
                // remote:192.168.1.50:9200:15-29
                stage.remoteHost = parts[1];
                stage.remotePort = std::stoi(parts[2]);
                std::string range = parts[3];
                size_t dash = range.find('-');
                if (dash != std::string::npos) {
                    stage.startLayer = std::stoi(range.substr(0, dash));
                    stage.endLayer = std::stoi(range.substr(dash + 1)) + 1;
                }
            }
            stage.remoteClient = std::make_shared<net::TenzoClient>(stage.remoteHost, stage.remotePort);
        }

        if (stage.startLayer < 0 || stage.endLayer > totalLayers || stage.startLayer >= stage.endLayer) {
            clearStages();
            return false;
        }

        stages.push_back(stage);
    }

    return !stages.empty();
}

void HeteroPipeline::autoPartition(int totalLayers, bool hasVulkan, bool hasCuda, bool hasRocm, const std::string& remoteNode) {
    clearStages();
    if (totalLayers <= 0) return;

    if (!remoteNode.empty()) {
        if (hasVulkan || hasCuda || hasRocm) {
            // 3-way partition: GPU -> CPU -> Remote
            int gpuLayers = totalLayers / 3;
            int cpuLayers = totalLayers / 3;

            PipelineStage gpuStage;
            if (hasCuda) gpuStage.type = HeteroDeviceType::GPU_CUDA;
            else if (hasRocm) gpuStage.type = HeteroDeviceType::GPU_ROCM;
            else gpuStage.type = HeteroDeviceType::GPU_VULKAN;
            gpuStage.startLayer = 0;
            gpuStage.endLayer = gpuLayers;
            stages.push_back(gpuStage);

            PipelineStage cpuStage;
            cpuStage.type = HeteroDeviceType::CPU;
            cpuStage.startLayer = gpuLayers;
            cpuStage.endLayer = gpuLayers + cpuLayers;
            stages.push_back(cpuStage);

            PipelineStage remoteStage;
            remoteStage.type = HeteroDeviceType::REMOTE_NODE;
            remoteStage.startLayer = gpuLayers + cpuLayers;
            remoteStage.endLayer = totalLayers;
            size_t colon = remoteNode.find(':');
            if (colon != std::string::npos) {
                remoteStage.remoteHost = remoteNode.substr(0, colon);
                remoteStage.remotePort = std::stoi(remoteNode.substr(colon + 1));
            } else {
                remoteStage.remoteHost = remoteNode;
                remoteStage.remotePort = 9200;
            }
            remoteStage.remoteClient = std::make_shared<net::TenzoClient>(remoteStage.remoteHost, remoteStage.remotePort);
            stages.push_back(remoteStage);
            return;
        } else {
            // Distributed mode: Split 50/50 between local CPU and remote
            int mid = totalLayers / 2;
            
            PipelineStage localStage;
            localStage.type = HeteroDeviceType::CPU;
            localStage.startLayer = 0;
            localStage.endLayer = mid;
            stages.push_back(localStage);

            PipelineStage remoteStage;
            remoteStage.type = HeteroDeviceType::REMOTE_NODE;
            remoteStage.startLayer = mid;
            remoteStage.endLayer = totalLayers;
            
            size_t colon = remoteNode.find(':');
            if (colon != std::string::npos) {
                remoteStage.remoteHost = remoteNode.substr(0, colon);
                remoteStage.remotePort = std::stoi(remoteNode.substr(colon + 1));
            } else {
                remoteStage.remoteHost = remoteNode;
                remoteStage.remotePort = 9200;
            }
            remoteStage.remoteClient = std::make_shared<net::TenzoClient>(remoteStage.remoteHost, remoteStage.remotePort);
            stages.push_back(remoteStage);
            return;
        }
    }

    if (hasVulkan || hasCuda || hasRocm) {
        // Hybrid CPU + GPU: 60% of layers on GPU, 40% on CPU
        int gpuLayers = (totalLayers * 6) / 10;
        if (gpuLayers > 0) {
            PipelineStage gpuStage;
            if (hasCuda) gpuStage.type = HeteroDeviceType::GPU_CUDA;
            else if (hasRocm) gpuStage.type = HeteroDeviceType::GPU_ROCM;
            else gpuStage.type = HeteroDeviceType::GPU_VULKAN;
            
            gpuStage.startLayer = 0;
            gpuStage.endLayer = gpuLayers;
            stages.push_back(gpuStage);
        }

        if (gpuLayers < totalLayers) {
            PipelineStage cpuStage;
            cpuStage.type = HeteroDeviceType::CPU;
            cpuStage.startLayer = gpuLayers;
            cpuStage.endLayer = totalLayers;
            stages.push_back(cpuStage);
        }
    } else {
        // CPU only
        PipelineStage cpuStage;
        cpuStage.type = HeteroDeviceType::CPU;
        cpuStage.startLayer = 0;
        cpuStage.endLayer = totalLayers;
        stages.push_back(cpuStage);
    }
}

void HeteroPipeline::printTopology() const {
    std::cout << "[HeteroPipeline] Active Execution Stages (" << stages.size() << " stages):\n";
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto& s = stages[i];
        std::cout << "  ├─ Stage " << i << ": ";
        switch (s.type) {
            case HeteroDeviceType::CPU:
                std::cout << "CPU (Host SIMD)";
                break;
            case HeteroDeviceType::GPU_VULKAN:
                std::cout << "Vulkan GPU (Device " << s.deviceIndex << ")";
                break;
            case HeteroDeviceType::GPU_CUDA:
                std::cout << "CUDA GPU (Device " << s.deviceIndex << ")";
                break;
            case HeteroDeviceType::GPU_ROCM:
                std::cout << "ROCm GPU (Device " << s.deviceIndex << ")";
                break;
            case HeteroDeviceType::REMOTE_NODE:
                std::cout << "Remote Node (" << s.remoteHost << ":" << s.remotePort << ")";
                break;
        }
        std::cout << " => Layers [" << s.startLayer << ".." << (s.endLayer - 1) << "] ("
                  << (s.endLayer - s.startLayer) << " layers)\n";
    }
}

} // namespace tenzo
