# CI emulation

> [!NOTE]
> Everything in `emulation/` is temporary. It stands in for the parts of the CI that
> can't run yet: publishing to the Hub and loading the published kernel back. Delete
> this folder once [`publish.yml`](../.github/workflows/publish.yml) runs for real.
> Nothing outside this folder refers to it, and CI doesn't run its tests; run them
> locally with `python -m pytest emulation/tests`.

`emulation/ci_emulate.py` runs this repository's CI locally, kernel by kernel, for the
kernels in [`approved-kernels.toml`](approved-kernels.toml). It runs the real pull
request stages from [`scripts/kernel_ci.py`](../scripts/kernel_ci.py), then **emulates
publishing**: kernels are written to a local copy of the Hub's layout, never uploaded.
`get_kernel()` is then checked against that local Hub exactly as users will call it
against the real one.

| Stage       | What it does                                                                                                                                                                | Proves                                                         | From                   |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------- |
| `validate`  | `scripts/check_wrapping.py` (+ `kernel-builder check-config` with Nix)                                                                                                      | the kernel is wrapped correctly for Hugging Face's build logic | `scripts/kernel_ci.py` |
| `build`     | `nix build .#backendBundle.<backend>` (all variants; `--ci` for one), then `kernel-builder check-abi`; or `--build-dir` for a prebuilt build                                | it builds with kernel-builder                                  | `scripts/kernel_ci.py` |
| `test`      | `nix build .#ci-test`, then `ci-test` on the AMD GPU (`pytest -m kernels_ci` with `LOCAL_KERNELS`)                                                                          | it works on the hardware                                       | `scripts/kernel_ci.py` |
| `publish`   | **emulated**: `emulation/local_hub.py` copies `build/` into `<out>/hub`, branch `v<version>` (and `main` for version 1)                                                     | what would be uploaded                                         | emulation              |
| `verify`    | `scripts/ci_verify_load.py` in a user-like Python env: `get_kernel()` from the emulated Hub (`HF_HUB_OFFLINE=1`, `KERNELS_CACHE`), then the kernel's own `kernels_ci` tests | users can load and use the published kernel                    | emulation              |
| `benchmark` | times the `Benchmark` classes in `benchmarks/benchmark.py` from the emulated Hub; optional baseline comparison                                                              | performance, recorded in `perf.json`                           | emulation              |

Results go to `ci-out/<run-id>/`: `summary.md`, `summary.json`, and per kernel
`findings.json`, logs, `upload.json`, `verify.json` and `perf.json`.

## What's here

| File                    | Purpose                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| `ci_emulate.py`         | the emulated pipeline and the self-test                                                                 |
| `local_hub.py`          | the emulated Hub: writes the cache layout `get_kernel()` reads offline                                  |
| `approved-kernels.toml` | kernels to run by default (empty). The real CI has no list: approval is a person approving each publish |
| `test-kernels.toml`     | AMD's existing `amd/*` kernels, as test cases                                                           |
| `tests/`                | tests for the above, and the real `kernels` loader against the emulated Hub                             |

The real CI uses `scripts/` (`kernel_ci.py`, `check_wrapping.py`,
`ci_verify_load.py`, …) and `ci/community.toml`, which stay.
`ci_verify_load.py --benchmark` is currently used only by the emulation's `benchmark`
stage.

## Running

```bash
# Tests (no GPU, Nix or network needed; loader tests need torch + kernels)
python -m pytest emulation/tests

# The approved kernels, all stages (on a machine with Nix and an AMD GPU)
python emulation/ci_emulate.py run --python /path/to/rocm-venv/bin/python

# Show the commands without running them
python emulation/ci_emulate.py run --dry-run

# Try a kernel from another directory, with Hugging Face's rules only (the approved list isn't read)
python emulation/ci_emulate.py run --kernels-root /path/to/kernels --kernel <kernel> --hf-rules-only

# Use a prebuilt build/ instead of Nix
python emulation/ci_emulate.py run --kernel <dir> --build-dir path/to/build

# Publish to a staging branch instead of v<version>
python emulation/ci_emulate.py run --branch staging

# Compare performance with an earlier run
python emulation/ci_emulate.py run --baseline ci-out/<earlier-run>/<kernel>/perf.json --fail-on-regression
```

`--python` is the user-like environment for `verify` and `benchmark`: PyTorch for ROCm
plus `pip install kernels pytest`, no Nix. It defaults to the interpreter running the
script.

For just the pull request stages, as CI runs them, use `scripts/kernel_ci.py` directly:

```bash
python scripts/kernel_ci.py run --kernel <kernel> --ci --fail-on-skip
```

## How it maps to the real CI

- [`kernel-ci.yml`](../.github/workflows/kernel-ci.yml) (every pull request) runs
  `scripts/kernel_ci.py`: `validate` on a GitHub-hosted runner, and
  `validate,build,test --ci --fail-on-skip` on the MI350
  ([`rocm-build-test.yml`](../.github/workflows/rocm-build-test.yml)).
- [`publish.yml`](../.github/workflows/publish.yml) (on request) replaces this folder's
  `publish` and `verify`: the same build with every variant, then a real, approved
  upload, then `ci_verify_load.py` against the real Hub, with `--tests-repo-id` so the
  kernel's tests follow a publish to the test organisation.

## Test cases: AMD's existing kernels

[`test-kernels.toml`](test-kernels.toml) lists the sources of AMD's existing `amd/*`
Hub packages (`aiter-kernels`, `aiter-flash-attn`,
`aiter-sage-attention`, `aiter-rope`; the Composable Kernel package is left out). They
exercise the CI on real, published kernels:

```bash
python emulation/ci_emulate.py run --approved emulation/test-kernels.toml --kernels-root /path/to/kernels
```

The list applies Hugging Face's wrapping rules only (`rules = "hf"`) and records known
results: `aiter-flash-attn` is expected to fail validation, because none of its tests is
marked `kernels_ci`, and its later stages still run (`keep_going`). Results that match
`expect` don't fail the run. GPU stages skip automatically when the machine's GPU isn't one
of the kernel's `rocm-archs`, e.g. `aiter-sage-attention` (gfx1201 only) on an MI350.
`emulation/tests/amd_test_cases_test.py` checks the list against the checkout when
`TEST_KERNELS_ROOT` points at it.

## Checking the emulation against the real Hub

```bash
python emulation/ci_emulate.py selftest --reference <org>/<kernel> --version N
```

The self-test downloads a real Hub kernel (read-only), re-publishes the same files to
the emulated Hub, loads it from both, and checks they match: same build variant, same
module file, same result, digest verified, same cache layout. It is the only step that
needs network access.

## Nothing is uploaded

No script here or in `scripts/` calls `kernel-builder upload`, `hf upload` or a Hugging
Face Hub write API (`emulation/tests/no_upload_test.py` checks this). `HF_TOKEN` and
`HUGGING_FACE_HUB_TOKEN` are removed from every command's environment, and every stage
runs with `HF_HUB_OFFLINE=1`. Emulated uploads are recorded as unsigned.

The only real upload is in [`publish.yml`](../.github/workflows/publish.yml), behind an
approval, in a job that runs no kernel code.

## Known gaps, to check on the first GPU run

- How the Nix `ci-test` reaches the ROCm driver. CUDA jobs symlink `libcuda` into
  `/run/opengl-driver/lib`; whether ROCm needs an equivalent is untested.
- The exact variants `.#backendBundle.rocm` produces (`summary.md` lists them).
- `benchmark` uses its own timing loop over the `Benchmark` class rather than
  `kernels benchmark`, so its numbers can differ from Hugging Face's benchmark pages.
