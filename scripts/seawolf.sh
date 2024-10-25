build() {
    cargo build
}

build-jetson() {
    cd /home/aquapack/SW8S-Rust/jetson/ && cargo run --all-features
}

run() {
    cargo run
}

help() {
    echo "Usage:"
    echo "  build:"
    echo "      \"\"                -- Builds locally using cargo"
    echo "      jetson              -- Cross compiles for the Jetson (arm64)"
    echo "  run                 -- Runs an arbitrary command on the remote device "
    echo "  "
    echo ""
    echo "Usage: seawolf (build | run)"
}

case $1 in 
    "run")
        run
    ;;
    "build")
        case $2 in
            "jetson")
                build-jetson
            ;;
            "")
                build
            ;;
            *)
                echo "Unknown option: \"$2\""
                help
            ;;
        esac
    ;;
    *)
        echo "Unknown option: \"$1\""
        help
        ;;
esac