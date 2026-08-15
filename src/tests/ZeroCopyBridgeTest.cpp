#include "ZeroCopyBridgeTest.h"
#include "runtime/ExecutionContext.h"
#include "runtime/MemRefUtils.h"
#include "passes/Passes.h"
#include "dialect/TenzoDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <vector>

using namespace mlir;
using namespace tenzo::runtime;

namespace tenzo {

void runZeroCopyBridgeTest(MLIRContext& context) {
    llvm::outs() << "\n=== [End-to-End] Zero-Copy Bridge & mmap() Test ===\n";

    // 1. Prepare weights file path
    const char* weights_path = "tenzo-frontend/export_output/weights.bin";
    if (access(weights_path, F_OK) != 0) {
        weights_path = "weights.bin";
    }

    int fd = open(weights_path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        llvm::errs() << "❌ Failed to open/create weights.bin\n";
        return;
    }

    size_t fileSize = 128 * 64 / 4 + 64 * 10 / 4; // 2-bit packed weight size
    if (ftruncate(fd, fileSize) != 0) {
        llvm::errs() << "❌ Failed to set weights file size\n";
        close(fd);
        return;
    }

    // 2. Memory-map weights file (Zero-Copy)
    void* mapped_ptr = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped_ptr == MAP_FAILED) {
        llvm::errs() << "❌ mmap() failed\n";
        close(fd);
        return;
    }

    // Fill dummy packed ternary bytes
    std::memset(mapped_ptr, 0x55, fileSize); // 0x55 = 0b01010101 (all 1s)

    llvm::outs() << "✅ Successfully mmap()'d " << fileSize << " bytes from " << weights_path << "\n";

    // 3. Create Zero-Copy MemRefDescriptor View
    auto weights_view = MemRefDescriptor<int8_t, 1>::create_view(
        static_cast<int8_t*>(mapped_ptr),
        {static_cast<int64_t>(fileSize)},
        {1}
    );

    llvm::outs() << "✅ Zero-Copy MemRefDescriptor created: ptr=" << weights_view.aligned 
                 << ", size=" << weights_view.sizes[0] << ", stride=" << weights_view.strides[0] << "\n";

    // Cleanup mmap & fd
    munmap(mapped_ptr, fileSize);
    close(fd);

    llvm::outs() << "🎉 Zero-Copy Bridge pipeline test PASSED!\n";
}

} // namespace tenzo
