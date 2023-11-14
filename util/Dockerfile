# Jetpack 4.6.4 is the last supported l4t for Jetson Nano
FROM timongentzsch/l4t-ubuntu20-crosscompile:latest

# Install system packages
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		build-essential \
		llvm \
		libclang-dev \
		clang \
		wget \
		libgstreamer1.0-dev \
		libgstreamer-plugins-base1.0-dev \
		libavutil-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libavresample-dev \
		libopenexr-dev \
		liblapack-dev \
		libatlas-base-dev \
		libdc1394-22-dev \
		libfreetype-dev \
		libharfbuzz-dev \
		libcudnn8

# Install OpenCV build
ADD https://github.com/ncsurobotics/SW8S-OpenCVBuild/releases/download/4.6.0-build2/linux-arm64-jetsonnano.sh .
RUN chmod +x linux-arm64-jetsonnano.sh && \
	./linux-arm64-jetsonnano.sh --prefix=/ --exclude-subdir && \
	rm linux-arm64-jetsonnano.sh
RUN printf '%s\n' 'source /opt/opencv-4.6.0/bin/setup_vars_opencv4.sh' >> ~/.bashrc

# Cargo/Rust setup
ENV CARGO_TERM_PROGRESS_WHEN=always
ENV CARGO_TERM_PROGRESS_WIDTH=80
ADD --chmod=100 https://sh.rustup.rs rust_install.sh
RUN ./rust_install.sh -y && \
	bash -c 'chmod +x "${HOME}/.cargo/env"' && \
	rm rust_install.sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Update dynamic linkage
RUN ldconfig

# Pull Rust dependencies
COPY --link Cargo.* .
RUN --mount=type=cache,target=/root/.cargo/registry ["cargo", "fetch"]

# Build dir for SW8S-Rust
RUN mkdir /build
WORKDIR /build

# Run build by default
ENTRYPOINT ["cargo", "build", "--color", "always", "--target-dir", "jetson_target"]
