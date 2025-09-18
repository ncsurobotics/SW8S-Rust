{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {
        pkgs,
        config,
        ...
      }: {
        devShells = let
          rustToolchain = with inputs.fenix.packages.${pkgs.system};
            combine (
              with stable; [
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
            cargo-expand
            cargo-info
            rust-analyzer
            bacon
          ];
        in {
          default =
            (pkgs.buildFHSEnv {
              name = "sw8s-rust-fhs";
              targetPkgs = pkgs:
                with pkgs.llvmPackages_19;
                  [
                    libclang
                    libllvm
                    bintools
                    clang
                  ]
                  ++ rustPackages;
              extraOutputsToInstall = [
                "dev"
                "out"
              ];
              profile = ''
                export JETSON_DEVSHELL_MODE=1
              '';
            }).env;
          noFHS = pkgs.mkShell {
            nativeBuildInputs = [pkgs.pkg-config];
            buildInputs = with pkgs;
              [
                llvmPackages_19.clang
                opencv
              ]
              ++ rustPackages;
            LIBCLANG_PATH = "${pkgs.llvmPackages_19.libclang.lib}/lib";
          };
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
