{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          venvDir = ".venv";
          packages = let
            customOpenCV = rec {
              opencv4 = pkgs.python312Packages.opencv4.override {
                enableGtk2 = true;
                gtk2 = pkgs.gtk2;
              };
              opencv-python = pkgs.python312Packages.opencv-python.override {
                inherit opencv4;
              };
            };
          in with pkgs; [ gtk2 python312 ] ++
            (with pkgs.gst_all_1; [
              gstreamer
              gst-plugins-base
              gst-plugins-good
              gst-plugins-bad
              gst-plugins-ugly
              gst-libav
              gst-vaapi
              gst-rtsp-server
            ]) ++
            (with pkgs.python312Packages; [
              pip
              numpy
              matplotlib
              ipywidgets
              nicegui
              # pkgconfig
              # opencv-python
              venvShellHook
              python-lsp-ruff
              python-lsp-server
              ultralytics
              onnxruntime
              pyside6
            ]) ++ [ customOpenCV.opencv-python ];
        };
      });
    };
}
