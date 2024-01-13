#!/usr/bin/env sh
# Original written by Marcus Behel
# Source https://github.com/MB3hel/RustCrossExperiments

################################################################################
# Setup
################################################################################
# Default value if not specified via env var
# $JETSON_SYSROOT should be an absolute path
[ -z "$JETSON_SYSROOT" ] && JETSON_SYSROOT="${PWD}/sysroot-jetson"

if [ ! -d "$JETSON_SYSROOT" ]; then
    echo "Sysroot $JETSON_SYSROOT does not exist. Specify using JETSON_SYSROOT variable."
    exit 1
fi
################################################################################


################################################################################
# Flags passed to clang
################################################################################
# Passed to everything (c, c++, linker)
shared_flags="-target aarch64-linux-gnu"                                                            # Tells clang to compile for aarch64 linux
shared_flags="$shared_flags -mcpu=cortex-a57"                                                       # Jetson CPU
shared_flags="$shared_flags -fuse-ld=lld"                                                           # Use lld b/c it supports -target
shared_flags="$shared_flags --sysroot=$JETSON_SYSROOT"                                              # Specify sysroot
shared_flags="$shared_flags -L$JETSON_SYSROOT/usr/local/cuda-10.2/targets/aarch64-linux/lib/"       # CUDA is here on jetson sysroot
shared_flags="$shared_flags -L$JETSON_SYSROOT/opt/opencv-4.6.0/lib/"                                # Custom OpenCV is here on jetson sysroot

# Only to clang to compile C code
cflags="$shared_flags"

# Only to clant++ to compile C++ code
cxxflags="$shared_flags"

# To linker (and rustflags as link-args)
ldflags="$shared_flags"
################################################################################


################################################################################
# Environment setup
################################################################################
# Make sure any C/C++ code built by crates uses right compilers / flags
# Note: Using triple specific vars so that tools built for build system as a
# part of the build process build as intended.
# Note that these should have target triple lower case unlike vars for cargo
export CC_aarch64_unknown_linux_gnu=clang
export CXX_aarch64_unknown_linux_gnu=clang++
export AR_aarch64_unknown_linux_gnu=llvm-ar
export CFLAGS_aarch64_unknown_linux_gnu="$cflags"
export CXXFLAGS_aarch64_unknown_linux_gnu="$cxxflags"
export LDFLAGS_aarch64_unknown_linux_gnu="$ldflags"
################################################################################


################################################################################
# Cargo flags / tools setup for target
################################################################################
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=clang
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_AR=llvm-ar
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS=""
for arg in $(echo $ldflags | sed 's/ /\n/g'); do
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="$CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS -C link-args=$arg"
done
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS#?}"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS
################################################################################


################################################################################
# OpenCV stuff
################################################################################
# Ensures correct OpenCV found (using environment var search method)
OPENCV_LINK_LIBS=""
for lib in $(find $JETSON_SYSROOT/opt/opencv-4.6.0/lib/ -maxdepth 1 -name "*.so"); do
    lib_name=$(basename $lib)
    lib_name=${lib_name#???}        # Remove first 3 chars ("lib")
    lib_name=${lib_name%???}        # Remove final 3 chars (".so")
    OPENCV_LINK_LIBS="$OPENCV_LINK_LIBS,$lib_name"
done
OPENCV_LINK_LIBS="${OPENCV_LINK_LIBS#?}"
export OPENCV_LINK_LIBS
export OPENCV_LINK_PATHS="$JETSON_SYSROOT/opt/opencv-4.6.0/lib/"
export OPENCV_INCLUDE_PATHS="$JETSON_SYSROOT/opt/opencv-4.6.0/include/opencv4,$JETSON_SYSROOT/opt/opencv-4.6.0/include/opencv4/opencv2"
export OPENCV_DISABLE_PROBES="pkg_config,cmake,vcpkg_cmake,vcpkg"
################################################################################


################################################################################
# Run the build
################################################################################
rustup target add aarch64-unknown-linux-gnu
cargo "$@" --target aarch64-unknown-linux-gnu --target-dir target-jetson
################################################################################
