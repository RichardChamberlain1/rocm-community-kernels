{
  description = "Flake for AITER MoE (AMD ROCm) kernels";

  inputs = {
    # aiter/ is a git submodule (ROCm/aiter); the kernels are built from it.
    self.submodules = true;
    kernel-builder.url = "github:huggingface/kernels/torch-2.15.0-final";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
