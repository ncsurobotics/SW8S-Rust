#[cfg(feature = "graphing")]
mod graphing {
    use proc_macro2::Span;
    use quote::quote;
    use std::env::var_os;
    use std::ffi::OsStr;
    use std::fs::{create_dir_all, read_dir, read_to_string, write};
    use std::path::{Path, PathBuf};
    use syn::fold::{fold_item_fn, fold_use_path, Fold};

    use syn::{
        parse_file, Ident, ItemFn, ReturnType, Token, Type, TypeParamBound, UsePath, UseTree,
    };

    struct ReplaceActionExec<'a> {
        actions: &'a mut Vec<String>,
    }

    impl<'a> ReplaceActionExec<'a> {
        fn new(actions: &'a mut Vec<String>) -> Self {
            Self { actions }
        }
    }

    impl Fold for ReplaceActionExec<'_> {
        // Replace every impl ActionExec with impl Action
        // Uses GraphAction to always be sure Action is imported
        // Also strips generics
        fn fold_item_fn(&mut self, i: ItemFn) -> ItemFn {
            let mut j = i.clone();
            if let ReturnType::Type(_, ref mut ret_type) = j.sig.output {
                if let Type::ImplTrait(ref mut impl_trait) = **ret_type {
                    impl_trait.bounds.iter_mut().for_each(|bound| {
                        if let TypeParamBound::Trait(ref mut t) = bound {
                            t.path.segments.iter_mut().for_each(|seg| {
                                if seg.ident == "ActionExec" {
                                    seg.ident = Ident::new("GraphActionExec", seg.ident.span());

                                    if j.sig.inputs.len() == 1 {
                                        self.actions.push(j.sig.ident.to_string());
                                    }

                                    /*
                                    j.sig.generics.type_params_mut().for_each(|param| {
                                        if param.ident == "Con" {
                                            param.bounds = Punctuated::new()
                                        }
                                    });
                                    */
                                }
                            });
                        }
                    });
                }
            };
            // Call on the regular recursion
            fold_item_fn(self, j)
        }

        // Replace local paths with global paths
        fn fold_use_path(&mut self, i: syn::UsePath) -> syn::UsePath {
            println!("Folding path segment: {:?}", i.ident);
            let mut j = i.clone();
            match j.ident.to_string().as_str() {
                "crate" => j.ident = Ident::new("sw8s_rust_lib", j.ident.span()),
                "super" => {
                    j.ident = Ident::new("sw8s_rust_lib", j.ident.span());
                    j.tree = Box::new(UseTree::Path(UsePath {
                        ident: Ident::new("missions", Span::call_site()),
                        colon2_token: Token![::](Span::call_site()),
                        tree: j.tree,
                    }));
                }
                _ => (),
            }
            // Call on the regular recursion
            fold_use_path(self, j)
        }
    }

    fn get_files(file: PathBuf) -> Vec<PathBuf> {
        if file.is_file() {
            vec![file]
        } else if file.is_dir() {
            read_dir(file)
                .unwrap()
                .flat_map(|inner| get_files(inner.unwrap().path()))
                .collect()
        } else {
            vec![]
        }
    }

    pub fn graphing_variants() {
        // Command to cargo
        println!("cargo:rerun-if-changed=src/missions");

        // Calculating output dir
        let out_dir = var_os("OUT_DIR").unwrap().to_str().unwrap().to_string();
        println!("out_dir: {out_dir}");
        let out_path_raw = out_dir + "/graph_missions";
        let out_path = Path::new(&out_path_raw);

        // Get parseable file clones
        let mission_files = get_files("src/missions".into())
            .into_iter()
            .filter(|path| path.extension().unwrap_or(OsStr::new("")) == "rs")
            .map(|path| {
                println!("Path: {:?}", path);
                (
                    path.clone(),
                    parse_file(&read_to_string(path).unwrap()).unwrap().clone(),
                )
            });

        // Modify clones and write to output dir
        mission_files
            .map(|(path, file)| {
                let mut actions = vec![];
                (
                    path,
                    ReplaceActionExec::new(&mut actions).fold_file(file),
                    actions,
                )
            })
            .for_each(|(path, file, actions)| {
                let actions_str =
                    "pub fn graph_actions<T: GraphActionContext::GetMainElectronicsBoard + GraphActionContext::GetControlBoard<tokio::io::WriteHalf<tokio_serial::SerialStream>> + GraphActionContext::GetFrontCamMat + Send + Sync + std::marker::Unpin>(context: &T) -> Vec<(String, Box<dyn GraphAction + '_>)> { vec!["
                        .to_string()
                        + &actions
                            .into_iter()
                            .fold("".to_string(), |acc, x| acc + &format!("(\"{x}\".to_string(), Box::new({x}(context))),"))
                        + "]}";
                let file_contents =
                    quote! { use sw8s_rust_lib::missions::action::Action as GraphAction; use sw8s_rust_lib::missions::action::ActionExec as GraphActionExec; use sw8s_rust_lib::missions::action_context as GraphActionContext; #file };
                let output_loc = out_path.join(path.strip_prefix::<PathBuf>("src/missions".into()).unwrap());
                create_dir_all(output_loc.parent().unwrap()).unwrap();
                write(
                    out_path.join(path.strip_prefix::<PathBuf>("src/missions".into()).unwrap()),
                    format!("{} {}", file_contents, actions_str),
                )
                .unwrap()
            });
    }
}

fn main() {
    #[cfg(feature = "graphing")]
    graphing::graphing_variants();

    #[cfg(feature = "cuda")]
    {
        // Rebuild on any kernel change
        println!("cargo:rerun-if-changed=src/cuda_kernels");

        // Rebuild for specific files that use kernels changing
        println!("cargo:rerun-if-changed=src/vision/nn_cv2.rs");

        cc::Build::new()
            .cuda(true)
            .flag("-cudart=shared")
            // Specify all cuda kernels that need to be built
            .file("src/cuda_kernels/process_net.cu")
            .compile("libsw8s_cuda.a");

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
    }
}
