{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      perSystem =
        { pkgs, config, ... }:
        {
          devShells =
            let
              rustToolchain =
                with inputs.fenix.packages.${pkgs.system};
                combine (
                  with stable;
                  [
                    clippy
                    rustc
                    cargo
                    rustfmt
                    rust-src
                    targets.aarch64-unknown-linux-gnu.stable.rust-std
                  ]
                );
              rustPackages = with pkgs; [
                rustToolchain
                openssl
                pkg-config
                cargo-deny
                cargo-edit
                cargo-watch
                rust-analyzer
              ];
            in
            {
              default =
                (pkgs.buildFHSEnv.override { stdenv = pkgs.llvmPackages_19.libcxxStdenv; } {
                  name = "sw8s-rust-fhs";
                  targetPkgs =
                    pkgs: with pkgs; [
                      gccNGPackages_15.libstdcxx
                      llvmPackages_19.clang-tools
                      llvmPackages_19.libclang
                      llvmPackages_19.libllvm
                      gcc
                      glibc
                      glib
                      glm
                      llvmPackages_19.bintools
                      llvmPackages_19.clang
                      llvmPackages_19.libcxx
                    ] ++ rustPackages;
          					extraOutputsToInstall = ["dev" "out" "lib "];
          					profile = ''
                      export BINDGEN_EXTRA_CLANG_ARGS="-isystem /usr/lib/clang/19/include/"
                      export NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE -I /usr/lib/clang/19/include/"
                      export JETSON_DEVSHELL_MODE=1
        					  '';
                }).env;
            };
          formatter = pkgs.nixfmt-tree;
          packages.image = pkgs.dockerTools.streamNixShellImage {
            name = "test";
            tag = "latest";
            drv = config.devShells.fhs;
          };
        };
    };
}
