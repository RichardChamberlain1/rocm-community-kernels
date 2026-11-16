import kernels
import pytest
import torch

# The repo id and version from build.toml. In CI, LOCAL_KERNELS points this at the
# fresh build; never import the package directly.
bad_community = kernels.get_kernel("amd/bad-community", version=1)


# Deliberately no kernels_ci marker: CI's validate stage must reject this kernel.
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scale(dtype):
    x = torch.randn(10_000, device="cuda", dtype=dtype)
    torch.testing.assert_close(bad_community.scale(x, 2.0), x * 2.0)


def test_scale_empty():
    x = torch.empty(0, device="cuda")
    assert bad_community.scale(x, 2.0).numel() == 0
