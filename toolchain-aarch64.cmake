# =============================================================================
# CMake Toolchain File for aarch64 (ARM64) Cross-Compilation
# =============================================================================
#
# IMPORTANT: Cross-compilation requires aarch64 target libraries.
#
# Option 1: Install aarch64 libraries on host (Debian/Ubuntu multiarch)
#   sudo dpkg --add-architecture arm64
#   sudo apt update
#   sudo apt install libopencv-dev:arm64 libyaml-cpp-dev:arm64
#
# Option 2: Use a custom sysroot from RK3588 SDK or board
#   cmake -DCMAKE_TOOLCHAIN_FILE=toolchain-aarch64.cmake \
#         -DAARCH64_SYSROOT=/path/to/rk3588-sysroot ..
#
# Option 3: Build directly on the RK3588 board (recommended for simplicity)
#
# =============================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler configuration
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Sysroot configuration (optional - only if AARCH64_SYSROOT is provided)
if(DEFINED AARCH64_SYSROOT OR DEFINED ENV{AARCH64_SYSROOT})
    if(DEFINED ENV{AARCH64_SYSROOT} AND NOT DEFINED AARCH64_SYSROOT)
        set(AARCH64_SYSROOT $ENV{AARCH64_SYSROOT})
    endif()

    if(EXISTS "${AARCH64_SYSROOT}")
        set(CMAKE_SYSROOT ${AARCH64_SYSROOT})
        set(CMAKE_FIND_ROOT_PATH ${AARCH64_SYSROOT})
        message(STATUS "Cross-compiling with sysroot: ${AARCH64_SYSROOT}")

        # PKG_CONFIG configuration for cross-compilation
        set(ENV{PKG_CONFIG_PATH} "${AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig:${AARCH64_SYSROOT}/usr/lib/pkgconfig")
        set(ENV{PKG_CONFIG_LIBDIR} "${AARCH64_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig")
        set(ENV{PKG_CONFIG_SYSROOT_DIR} "${AARCH64_SYSROOT}")
    else()
        message(FATAL_ERROR "AARCH64_SYSROOT specified but path does not exist: ${AARCH64_SYSROOT}")
    endif()
else()
    # No sysroot specified - rely on multiarch or default paths
    message(STATUS "No AARCH64_SYSROOT specified. Using system multiarch paths.")
    message(STATUS "If build fails, install aarch64 libraries: apt install libopencv-dev:arm64 libyaml-cpp-dev:arm64")

    # Standard multiarch library paths
    set(CMAKE_FIND_ROOT_PATH
        "/usr/lib/aarch64-linux-gnu"
        "/usr/aarch64-linux-gnu"
    )
endif()

# Search rules for cross-compilation
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Disable sanitizers for cross-compilation (they require host runtime support)
set(ENABLE_SANITIZERS OFF CACHE BOOL "Disabled for cross-compilation" FORCE)

# OpenCV path hints for aarch64
list(APPEND CMAKE_PREFIX_PATH
    "/usr/lib/aarch64-linux-gnu/cmake/opencv4"
)
