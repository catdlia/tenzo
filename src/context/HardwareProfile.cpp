#include "HardwareProfile.h"
#include "X86HardwareProfile.h"
#include "ARMHardwareProfile.h"

namespace tenzo {
std::shared_ptr<HardwareProfile> HardwareProfile::detect() {
#ifdef __aarch64__
    return std::make_shared<ARMHardwareProfile>();
#else
    return std::make_shared<X86HardwareProfile>();
#endif
}
} // namespace tenzo
