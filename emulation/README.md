# CI emulation

> [!NOTE]
> Everything in `emulation/` is temporary. It runs the parts of the CI that can't run
> in GitHub yet: publishing to the Hub and loading the published kernel back. Delete
> this folder once [`publish.yml`](../.github/workflows/publish.yml) runs for real.
> Nothing outside this folder refers to it, and CI doesn't run its tests; run them
> locally with `python -m pytest emulation/tests`.

`emulation/ci_emulate.py` runs this repository's CI locally, kernel by kernel, for the
kernels in [`approved-kernels.toml`](approved-kernels.toml). It runs the real pull
request stages from [`scripts/kernel_ci.py`](../scripts/kernel_ci.py), then publishes,
in one of two ways:

- **Emulated** (default): kernels are written to a local copy of the Hub's layout,
  never uploaded. No network or token is needed.
- **Real, private** (`--hub-org rocm-community-poc`): kernels are uploaded to the
  proof-of-concept organisation [rocm-community-poc](https://huggingface.co/rocm-community-poc),
  always as private repositories (see [Publishing to the real Hub](#publishing-to-the-real-hub)).

`get_kernel()` is then checked against what was published, exactly as users will call it.

| Stage       | What it does                                                                                                                                                                 | Proves                                                         | From                   |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------- |
| `validate`  | `scripts/check_wrapping.py` (+ `kernel-builder check-config` with Nix)                                                                                                       | the kernel is wrapped correctly for Hugging Face's build logic | `scripts/kernel_ci.py` |
| `build`     | `nix build .#backendBundle.<backend>` (all variants; `--ci` for one), then `kernel-builder check-abi`; or `--build-dir` for a prebuilt build                                 | it builds with kernel-builder                                  | `scripts/kernel_ci.py` |
| `test`      | `nix build .#ciTests.<newest Torch with ROCm 7.1>`, then `ci-test` on the AMD GPU (`pytest -m kernels_ci` with `LOCAL_KERNELS`)                                              | it works on the hardware                                       | `scripts/kernel_ci.py` |
| `publish`   | emulated: `emulation/local_hub.py` copies `build/` into `<out>/hub`. Real: `kernel-builder upload --private` to `rocm-community-poc/<name>`, then a check that it's private  | what is uploaded, branch `v<version>` (and `main` for 1)       | emulation              |
| `verify`    | `scripts/ci_verify_load.py` in a user-like Python env: `get_kernel()` offline from the published kernel (`HF_HUB_OFFLINE=1`, `KERNELS_CACHE`), then its own `kernels_ci` tests | users can load and use the published kernel                    | emulation              |
| `benchmark` | times the `Benchmark` classes in `benchmarks/benchmark.py` from the published kernel; optional baseline comparison                                                           | performance, recorded in `perf.json`                           | emulation              |

Results go to `ci-out/<run-id>/`: `summary.md`, `summary.json`, and per kernel
`findings.json`, logs, `upload.json`, `verify.json` and `perf.json` (with the real Hub,
also `hub-upload-<branch>.json` and `download.json`).

## What's here

| File                    | Purpose                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| `ci_emulate.py`         | the emulated pipeline and the self-test                                                                 |
| `local_hub.py`          | the emulated Hub: writes the cache layout `get_kernel()` reads offline                                  |
| `hub_publish.py`        | the real Hub: private uploads to `rocm-community-poc` only, and the privacy check                       |
| `hub_delete.py`         | run by hand: shows POC repositories, and deletes one only after you type its id                         |
| `approved-kernels.toml` | kernels to run by default (empty). The real CI has no list: approval is a person approving each publish |
| `test-kernels.toml`     | the test kernels to run, with their expected results                                                    |
| `kernels/`              | the test kernels: three fixtures (see below)                                                            |
| `tests/`                | tests for the above, and the real `kernels` loader against the emulated Hub                             |

The real CI uses `scripts/` (`kernel_ci.py`, `check_wrapping.py`, `ci_download.py`,
`ci_verify_load.py`, `publish_target.py`, …) and `ci/community.toml`, which stay.
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

# Publish privately to rocm-community-poc: show the commands (no token, nothing sent)
python emulation/ci_emulate.py run --approved emulation/test-kernels.toml --hub-org rocm-community-poc --dry-run

# ... and for real (token from the environment; see below)
read -rs HF_TOKEN && export HF_TOKEN
python emulation/ci_emulate.py run --approved emulation/test-kernels.toml --only scale-triton --hub-org rocm-community-poc

# Delete POC repositories again, by hand: first look (page, visibility, branches) ...
python emulation/hub_delete.py --hub-org rocm-community-poc scale-triton
# ... check it on the Hub, then delete (asks you to type the full repository id)
python emulation/hub_delete.py --hub-org rocm-community-poc scale-triton --delete
```

`--python` is the user-like environment for `verify` and `benchmark`: PyTorch for ROCm
plus `pip install kernels pytest`, no Nix. It defaults to the interpreter running the
script.

Nix must use Hugging Face's binary cache, as the workflows do with `cachix-action`.
Without it, Nix builds every PyTorch × ROCm toolchain from source. Hugging Face's
`install.sh` sets the cache up; on a Nix you installed yourself, run
`nix run nixpkgs#cachix -- use huggingface`. On a Nix without a daemon, add
`extra-substituters = https://huggingface.cachix.org` and the matching
`extra-trusted-public-keys` to `/etc/nix/nix.custom.conf` (or `nix.conf`). `build` and
`test` fail early when the cache isn't configured (`--no-hf-cache-check` overrides
this). They also warn, before building, about any toolchain derivation (LLVM, clang,
rocm-clr, PyTorch) that the cache doesn't have and Nix will compile.

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
  `publish` and `verify`: the same build with every variant, then a real, approved,
  private upload, then `ci_download.py` with a read-only token and `ci_verify_load.py`
  offline without one, with `--tests-repo-id` so the kernel's tests follow a publish to
  the test organisation. `--hub-org rocm-community-poc` does the same here, except that
  approval is you running it, and nothing is signed.

## Test kernels

[`test-kernels.toml`](test-kernels.toml) lists the test kernels in
[`kernels/`](kernels/). They are test material only, removed with this folder. Kernel CI
only looks at top-level kernel directories, so it never sees them. With
`--hub-org rocm-community-poc`, the two that pass validate are published, privately.

```bash
python emulation/ci_emulate.py run --approved emulation/test-kernels.toml
```

| Kernel          | What it is                                                                                 | Expected                                                                            |
| --------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `scale-triton`  | Fixture from the `add-community-kernel` Triton template                                    | every stage passes                                                                  |
| `scale-hip`     | Fixture from the `add-community-kernel` HIP template                                       | every stage passes                                                                  |
| `bad-community` | `scale-triton` broken on purpose: reserved `amd` org, no card notice, no `kernels_ci` test | validate fails and every later stage is skipped: the gate stops it before the build |

The fixtures are the `scale(x, factor)` example from the skill's templates. They're not
performance-tuned and aren't meant to be real kernels: they exercise the CI end to end.
They run with the community rules from `ci/community.toml`, so they show those rules pass
a compliant kernel and stop a non-compliant one.

Results that match `expect` don't fail the run. GPU stages skip automatically when the
machine's GPU isn't one of the kernel's `rocm-archs`.
`emulation/tests/kernel_list_test.py` checks the list:

- every expectation holds;
- `bad-community` fails for exactly its planted reasons;
- the dry run stops `bad-community` at validate.

`TEST_KERNELS_ROOT` points it at another checkout. The fixtures' `flake.lock` files are
created by the first Nix build.

## Checking the emulation against the real Hub

```bash
python emulation/ci_emulate.py selftest --reference <org>/<kernel> --version N
```

The self-test downloads a real Hub kernel (read-only), re-publishes the same files to
the emulated Hub, loads it from both, and checks they match: same build variant, same
module file, same result, digest verified, same cache layout. Apart from
`--hub-org`, it is the only step that needs network access.

## Publishing to the real Hub

Without `--hub-org`, nothing is uploaded: no stage calls `kernel-builder upload` or a
Hugging Face Hub write API, `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are removed from
every command's environment, and every stage runs with `HF_HUB_OFFLINE=1`.

With `--hub-org rocm-community-poc`:

- **Only the POC organisation.** Any other `--hub-org` is refused, and so are names the
  community rules reserve for AMD (`aiter-`). The name comes from `build.toml` and the
  organisation is swapped by `scripts/publish_target.py`, as in `publish.yml`:
  `rocm-community/scale-triton` is published as `rocm-community-poc/scale-triton`.
- **Always private.** Every upload is `kernel-builder upload --private`. `--private` only
  applies when the upload creates the repository, so publish then checks that the
  repository is private and fails if it isn't.
- **Not signed.** Uploads are recorded as unsigned (`publish.yml` signs with cosign, whose
  public transparency log is permanent).
- **The token only where it's needed.** A real run reads `HF_TOKEN` from your
  environment and stops before any stage without it. It's passed, in the environment and
  never on the command line, to the upload and to `scripts/ci_download.py`, which
  downloads the kernel without loading it. `get_kernel()` and the tests then run offline
  from that download with no token and a fresh `HF_HOME`, so kernel code can't read a
  stored token. `--dry-run` reads no token and sends nothing.
- **Deleting is separate.** The emulation never deletes. `emulation/hub_delete.py`, run
  by hand, shows each repository (its Hub page, visibility, branches, last change) so you
  can inspect it on the Hub first. With `--delete` it asks you to type each full
  repository id, and deletes only on an exact match. Deleting is irreversible.

`emulation/hub_publish.py` is the only file here or in `scripts/` that uploads, and
`emulation/hub_delete.py`, which nothing imports, the only one that deletes
(`emulation/tests/hub_writes_test.py` checks both).

### The token

Use a fine-grained token (Hugging Face → Settings → Access Tokens) with write access to
`rocm-community-poc` repositories only; deleting needs admin rights in the organisation.
Never put a token in a file in this repository (`scripts/tests/no_secrets_test.py` and
the pull request gitleaks scan check). On the machine that runs the emulation:

```bash
read -rs HF_TOKEN && export HF_TOKEN   # paste the token; it isn't echoed or saved in history
# ... run the emulation ...
unset HF_TOKEN
```

Don't use `hf auth login` there: it saves the token in `~/.cache/huggingface/token`,
where kernel code run on that machine could read it.

## Known gaps, to check on the first GPU run

- How the Nix `ci-test` reaches the ROCm driver. CUDA jobs symlink `libcuda` into
  `/run/opengl-driver/lib`; whether ROCm needs an equivalent is untested.
- The exact variants `.#backendBundle.rocm` produces (`summary.md` lists them).
- What the binary cache covers. At kernel-builder `torch-2.15.0-final`, Hugging Face's
  cache has the ROCm build toolchain only for the ROCm 7.1 variants. Every ROCm 7.0, 7.2,
  7.14 and 10.0 variant compiles LLVM from source, Hugging Face's own `relu` included.
  Triton (`torch-noarch`) kernels don't compile it. The `test` stage therefore builds the
  ROCm 7.1 GPU test (`--ci-test-rocm`), not kernel-builder's `.#ci-test`, which is ROCm
  7.14. `backendCi.rocm` (`--ci`) is still ROCm 7.14, so a compiled kernel takes hours
  there, as does `backendBundle.rocm`, which builds every variant.
- `benchmark` uses its own timing loop over the `Benchmark` class rather than
  `kernels benchmark`, so its numbers can differ from Hugging Face's benchmark pages.
