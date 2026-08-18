#include "VulkanRuntime.h"
#include "VulkanShaders.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

#ifdef TENZO_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace tenzo {
namespace runtime {

bool VulkanRuntime::s_initialized = false;
std::string VulkanRuntime::s_deviceName = "Unknown";

#ifdef TENZO_HAS_VULKAN
static VkInstance s_instance = VK_NULL_HANDLE;
static VkPhysicalDevice s_physicalDevice = VK_NULL_HANDLE;
static VkDevice s_device = VK_NULL_HANDLE;
static VkQueue s_computeQueue = VK_NULL_HANDLE;
static VkCommandPool s_commandPool = VK_NULL_HANDLE;
static uint32_t s_computeQueueFamily = 0;

#define VK_CHECK(x) do { \
    VkResult err = x; \
    if (err != VK_SUCCESS) { \
        std::cerr << "[Vulkan] Error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while(0)

static uint32_t findComputeQueueFamily(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    }
    return 0;
}

static uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(s_physicalDevice, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((filter & (1 << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0;
}
#endif

bool VulkanRuntime::initialize() {
#ifdef TENZO_HAS_VULKAN
    if (s_initialized) return true;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Tenzo LLM Engine";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&instInfo, nullptr, &s_instance));

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(s_instance, &devCount, nullptr);
    if (devCount == 0) { 
        std::cerr << "[Vulkan] No GPU device found!\n"; 
        return false; 
    }

    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(s_instance, &devCount, devs.data());
    s_physicalDevice = devs[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(s_physicalDevice, &props);
    s_deviceName = props.deviceName;
    std::cout << "[Vulkan] GPU Compute Device: " << s_deviceName << "\n";

    s_computeQueueFamily = findComputeQueueFamily(s_physicalDevice);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = s_computeQueueFamily;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &prio;

    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    VK_CHECK(vkCreateDevice(s_physicalDevice, &devInfo, nullptr, &s_device));
    vkGetDeviceQueue(s_device, s_computeQueueFamily, 0, &s_computeQueue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = s_computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(s_device, &poolInfo, nullptr, &s_commandPool));

    s_initialized = true;
    return true;
#else
    return false;
#endif
}

void VulkanRuntime::cleanup() {
#ifdef TENZO_HAS_VULKAN
    if (!s_initialized) return;
    vkDestroyCommandPool(s_device, s_commandPool, nullptr);
    vkDestroyDevice(s_device, nullptr);
    vkDestroyInstance(s_instance, nullptr);
    s_initialized = false;
#endif
}

bool VulkanRuntime::isAvailable() { return s_initialized; }
std::string VulkanRuntime::getDeviceName() { return s_deviceName; }

bool VulkanRuntime::executeShader(
    const std::vector<uint32_t>& spirv,
    const std::vector<std::pair<void*, size_t>>& inputs,
    std::pair<void*, size_t> output,
    const void* pushConstantsData,
    size_t pushConstantsSize,
    std::array<uint32_t, 3> numWG
) {
#ifdef TENZO_HAS_VULKAN
    if (!s_initialized && !initialize()) return false;

    // 1. Shader module
    VkShaderModuleCreateInfo smInfo = {};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = spirv.size() * 4;
    smInfo.pCode = spirv.data();
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(s_device, &smInfo, nullptr, &shader));

    // 2. Descriptor layout
    uint32_t nBufs = static_cast<uint32_t>(inputs.size() + 1);
    std::vector<VkDescriptorSetLayoutBinding> bindings(nBufs);
    for (uint32_t i = 0; i < nBufs; i++) {
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo dslInfo = {};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = nBufs;
    dslInfo.pBindings = bindings.data();
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(s_device, &dslInfo, nullptr, &dsl));

    // 3. Pipeline layout with Push Constants
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = static_cast<uint32_t>(pushConstantsSize);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsl;
    if (pushConstantsSize > 0) {
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pcRange;
    }
    VkPipelineLayout pipeLayout;
    VK_CHECK(vkCreatePipelineLayout(s_device, &plInfo, nullptr, &pipeLayout));

    // 4. Compute pipeline
    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = shader;
    cpInfo.stage.pName = "main";
    cpInfo.layout = pipeLayout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(s_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline));

    // 5. Descriptor pool
    VkDescriptorPoolSize dpSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nBufs};
    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &dpSize;
    dpInfo.maxSets = 1;
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(s_device, &dpInfo, nullptr, &descPool));

    VkDescriptorSetAllocateInfo dsaInfo = {};
    dsaInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsaInfo.descriptorPool = descPool;
    dsaInfo.descriptorSetCount = 1;
    dsaInfo.pSetLayouts = &dsl;
    VkDescriptorSet descSet;
    VK_CHECK(vkAllocateDescriptorSets(s_device, &dsaInfo, &descSet));

    // 6. Create buffers
    std::vector<VkBuffer> bufs;
    std::vector<VkDeviceMemory> mems;
    auto mkBuf = [&](size_t sz, const void* data) {
        VkBufferCreateInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = sz;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VkBuffer b; vkCreateBuffer(s_device, &bi, nullptr, &b);
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(s_device, b, &mr);
        VkMemoryAllocateInfo mai = {};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m; vkAllocateMemory(s_device, &mai, nullptr, &m);
        vkBindBufferMemory(s_device, b, m, 0);
        if (data) { 
            void* p; 
            vkMapMemory(s_device, m, 0, sz, 0, &p); 
            memcpy(p, data, sz); 
            vkUnmapMemory(s_device, m); 
        }
        bufs.push_back(b); 
        mems.push_back(m);
    };
    for (auto& in : inputs) mkBuf(in.second, in.first);
    mkBuf(output.second, nullptr);

    // 7. Update descriptors
    std::vector<VkDescriptorBufferInfo> dbis(nBufs);
    std::vector<VkWriteDescriptorSet> wds(nBufs);
    for (uint32_t i = 0; i < nBufs; i++) {
        dbis[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        wds[i] = {}; 
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = descSet; 
        wds[i].dstBinding = i;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].descriptorCount = 1; 
        wds[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(s_device, nBufs, wds.data(), 0, nullptr);

    // 8. Command buffer
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = s_commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; 
    vkAllocateCommandBuffers(s_device, &cbai, &cmd);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &cbbi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
    if (pushConstantsSize > 0 && pushConstantsData) {
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(pushConstantsSize), pushConstantsData);
    }
    vkCmdDispatch(cmd, numWG[0], numWG[1], numWG[2]);
    vkEndCommandBuffer(cmd);

    // 9. Submit & Wait
    VkSubmitInfo si = {}; 
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; 
    si.pCommandBuffers = &cmd;
    VkFence fence; 
    VkFenceCreateInfo fi = {}; 
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(s_device, &fi, nullptr, &fence);
    vkQueueSubmit(s_computeQueue, 1, &si, fence);
    vkWaitForFences(s_device, 1, &fence, VK_TRUE, UINT64_MAX);

    // 10. Read back
    void* p; 
    vkMapMemory(s_device, mems.back(), 0, output.second, 0, &p);
    memcpy(output.first, p, output.second);
    vkUnmapMemory(s_device, mems.back());

    // Cleanup
    vkDestroyFence(s_device, fence, nullptr);
    vkFreeCommandBuffers(s_device, s_commandPool, 1, &cmd);
    vkDestroyDescriptorPool(s_device, descPool, nullptr);
    vkDestroyPipeline(s_device, pipeline, nullptr);
    vkDestroyPipelineLayout(s_device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(s_device, dsl, nullptr);
    vkDestroyShaderModule(s_device, shader, nullptr);
    for (size_t i = 0; i < bufs.size(); i++) {
        vkDestroyBuffer(s_device, bufs[i], nullptr);
        vkFreeMemory(s_device, mems[i], nullptr);
    }
    return true;
#else
    return false;
#endif
}

bool VulkanRuntime::executeBitLinearTL1(
    const float* x,
    const uint8_t* W_packed,
    float* y,
    uint32_t N,
    uint32_t K,
    float scale
) {
    struct BitLinearPC {
        uint32_t N;
        uint32_t K;
        float scale;
        float quant_scale;
    } pc = {N, K, scale, 1.0f};

    size_t x_bytes = K * sizeof(float);
    size_t w_bytes = (static_cast<size_t>(N) * K) / 4; // 2-bit packed
    size_t y_bytes = N * sizeof(float);

    std::vector<std::pair<void*, size_t>> inputs = {
        {(void*)x, x_bytes},
        {(void*)W_packed, w_bytes}
    };

    uint32_t wg_x = (N + 63) / 64;
    return executeShader(shaders::get_bitlinear_tl1_spirv(), inputs, {y, y_bytes}, &pc, sizeof(pc), {wg_x, 1, 1});
}

bool VulkanRuntime::executeGemmF32(
    const float* x,
    const float* W,
    float* y,
    uint32_t N,
    uint32_t K,
    float alpha
) {
    struct GemmPC {
        uint32_t N;
        uint32_t K;
        float alpha;
    } pc = {N, K, alpha};

    size_t x_bytes = K * sizeof(float);
    size_t w_bytes = static_cast<size_t>(N) * K * sizeof(float);
    size_t y_bytes = N * sizeof(float);

    std::vector<std::pair<void*, size_t>> inputs = {
        {(void*)x, x_bytes},
        {(void*)W, w_bytes}
    };

    uint32_t wg_x = (N + 63) / 64;
    return executeShader(shaders::get_gemm_f32_spirv(), inputs, {y, y_bytes}, &pc, sizeof(pc), {wg_x, 1, 1});
}

bool VulkanRuntime::executeRMSNorm(
    const float* x,
    const float* w,
    float* y,
    uint32_t dim,
    float eps
) {
    struct RMSNormPC {
        uint32_t dim;
        float eps;
    } pc = {dim, eps};

    size_t bytes = dim * sizeof(float);
    std::vector<std::pair<void*, size_t>> inputs = {
        {(void*)x, bytes},
        {(void*)w, bytes}
    };

    return executeShader(shaders::get_rmsnorm_spirv(), inputs, {y, bytes}, &pc, sizeof(pc), {1, 1, 1});
}

} // namespace runtime
} // namespace tenzo
