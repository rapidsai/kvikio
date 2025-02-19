# kvikio 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- Continue to make KvikIO a shared library by moving code from hpp to cpp ([#581](https://github.com/rapidsai/kvikio/pull/581)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)

## 🐛 Bug Fixes

- Set `POSITION_INDEPENDENT_CODE` on libcurl ([#625](https://github.com/rapidsai/kvikio/pull/625)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- allow deselecting nvcomp wheels ([#589](https://github.com/rapidsai/kvikio/pull/589)) [@jameslamb](https://github.com/jameslamb)
- Fix the misplaced libcurl in CMake ([#587](https://github.com/rapidsai/kvikio/pull/587)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)

## 🚀 New Features

- Remove upper bounds on cuda-python to allow 12.6.2 and 11.8.5 ([#564](https://github.com/rapidsai/kvikio/pull/564)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Use `rapids-pip-retry` in CI jobs that might need retries ([#619](https://github.com/rapidsai/kvikio/pull/619)) [@gforsyth](https://github.com/gforsyth)
- Use std::string ([#617](https://github.com/rapidsai/kvikio/pull/617)) [@madsbk](https://github.com/madsbk)
- Revert CUDA 12.8 shared workflow branch changes ([#609](https://github.com/rapidsai/kvikio/pull/609)) [@vyasr](https://github.com/vyasr)
- Build and test with CUDA 12.8.0 ([#600](https://github.com/rapidsai/kvikio/pull/600)) [@bdice](https://github.com/bdice)
- Separate more interface and definition. Add comments on std::future. Mark noexcept to compat mode-related functions ([#588](https://github.com/rapidsai/kvikio/pull/588)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Use GCC 13 in CUDA 12 conda builds. ([#586](https://github.com/rapidsai/kvikio/pull/586)) [@bdice](https://github.com/bdice)
- Zarr: pin to version 2.* ([#583](https://github.com/rapidsai/kvikio/pull/583)) [@madsbk](https://github.com/madsbk)
- Continue to make KvikIO a shared library by moving code from hpp to cpp ([#581](https://github.com/rapidsai/kvikio/pull/581)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Check if nightlies have succeeded recently enough ([#579](https://github.com/rapidsai/kvikio/pull/579)) [@vyasr](https://github.com/vyasr)
- Update cuda-python lower bounds to 12.6.2 / 11.8.5 ([#573](https://github.com/rapidsai/kvikio/pull/573)) [@bdice](https://github.com/bdice)
- Improve compatibility mode consistency ([#567](https://github.com/rapidsai/kvikio/pull/567)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Require approval to run CI on draft PRs ([#563](https://github.com/rapidsai/kvikio/pull/563)) [@bdice](https://github.com/bdice)
- Add breaking change workflow trigger ([#536](https://github.com/rapidsai/kvikio/pull/536)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add Initial Java Support for GDS to KvikIO ([#396](https://github.com/rapidsai/kvikio/pull/396)) [@aslobodaNV](https://github.com/aslobodaNV)

# kvikio 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- Use curl &gt;=8.5.0 to align with conda-forge and avoid CVEs. ([#574](https://github.com/rapidsai/kvikio/pull/574)) [@bdice](https://github.com/bdice)
- cufile version ([#565](https://github.com/rapidsai/kvikio/pull/565)) [@madsbk](https://github.com/madsbk)
- Add a new KvikIO compatibility mode &quot;AUTO&quot; ([#547](https://github.com/rapidsai/kvikio/pull/547)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Build KvikIO as a shared library ([#527](https://github.com/rapidsai/kvikio/pull/527)) [@madsbk](https://github.com/madsbk)
- Small improvements ([#493](https://github.com/rapidsai/kvikio/pull/493)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)

## 🐛 Bug Fixes

- Use curl &gt;=8.5.0 to align with conda-forge and avoid CVEs. ([#574](https://github.com/rapidsai/kvikio/pull/574)) [@bdice](https://github.com/bdice)
- cufile version ([#565](https://github.com/rapidsai/kvikio/pull/565)) [@madsbk](https://github.com/madsbk)
- Fix the pytest error for async io ([#559](https://github.com/rapidsai/kvikio/pull/559)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- fix library-loading issues in editable installs ([#553](https://github.com/rapidsai/kvikio/pull/553)) [@jameslamb](https://github.com/jameslamb)
- Backport `FindCUDAToolkit` from CMake 3.31 ([#550](https://github.com/rapidsai/kvikio/pull/550)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix exporting of include directories ([#540](https://github.com/rapidsai/kvikio/pull/540)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Moving details in file_handle.hpp to .cpp ([#539](https://github.com/rapidsai/kvikio/pull/539)) [@madsbk](https://github.com/madsbk)
- Disallow cuda-python 12.6.1 and 11.8.4 ([#537](https://github.com/rapidsai/kvikio/pull/537)) [@bdice](https://github.com/bdice)
- Fix case of find_package call ([#534](https://github.com/rapidsai/kvikio/pull/534)) [@vyasr](https://github.com/vyasr)
- CurlHandle: fix error message handling ([#522](https://github.com/rapidsai/kvikio/pull/522)) [@madsbk](https://github.com/madsbk)
- Don&#39;t use macros for cuda driver functions ([#516](https://github.com/rapidsai/kvikio/pull/516)) [@Jacobfaib](https://github.com/Jacobfaib)
- Fix CUDA driver type stub definitions ([#511](https://github.com/rapidsai/kvikio/pull/511)) [@Jacobfaib](https://github.com/Jacobfaib)
- Stop tagging wheels as arch-agnostic ([#507](https://github.com/rapidsai/kvikio/pull/507)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Include &lt;utility&gt; since we use std::exchange ([#505](https://github.com/rapidsai/kvikio/pull/505)) [@robertmaynard](https://github.com/robertmaynard)
- Disabling curl tests doesn&#39;t now disable kvikio cpp tests ([#503](https://github.com/rapidsai/kvikio/pull/503)) [@robertmaynard](https://github.com/robertmaynard)
- Visibility of static class methods ([#492](https://github.com/rapidsai/kvikio/pull/492)) [@madsbk](https://github.com/madsbk)

## 📖 Documentation

- to ([#532](https://github.com/rapidsai/kvikio/pull/532)) [@RichardScottOZ](https://github.com/RichardScottOZ)

## 🚀 New Features

- Add a new KvikIO compatibility mode &quot;AUTO&quot; ([#547](https://github.com/rapidsai/kvikio/pull/547)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Upgrade nvcomp to 4.1.0.6 ([#525](https://github.com/rapidsai/kvikio/pull/525)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- prefer wheel-provided libkvikio.so, use RTLD_LOCAL ([#551](https://github.com/rapidsai/kvikio/pull/551)) [@jameslamb](https://github.com/jameslamb)
- enforce wheel size limits, README formatting in CI ([#548](https://github.com/rapidsai/kvikio/pull/548)) [@jameslamb](https://github.com/jameslamb)
- remove WheelHelpers.cmake ([#545](https://github.com/rapidsai/kvikio/pull/545)) [@jameslamb](https://github.com/jameslamb)
- Put a ceiling on cuda-python ([#543](https://github.com/rapidsai/kvikio/pull/543)) [@jameslamb](https://github.com/jameslamb)
- Replace FindcuFile with upstream FindCUDAToolkit support ([#542](https://github.com/rapidsai/kvikio/pull/542)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Build KvikIO as a shared library ([#527](https://github.com/rapidsai/kvikio/pull/527)) [@madsbk](https://github.com/madsbk)
- print sccache stats in builds ([#524](https://github.com/rapidsai/kvikio/pull/524)) [@jameslamb](https://github.com/jameslamb)
- RemoteIO: use a pinned bounce buffer ([#519](https://github.com/rapidsai/kvikio/pull/519)) [@madsbk](https://github.com/madsbk)
- Use registered strings for NVTX. Add more NVTX annotations. ([#518](https://github.com/rapidsai/kvikio/pull/518)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Minor README fixes ([#517](https://github.com/rapidsai/kvikio/pull/517)) [@Jacobfaib](https://github.com/Jacobfaib)
- Python bindings to `cuFileDriverOpen()` and `cuFileDriverClose()` ([#514](https://github.com/rapidsai/kvikio/pull/514)) [@madsbk](https://github.com/madsbk)
- Add CUDA libs in Python Conda, Consolidate Conda CI installs &amp; use `rapids-dask-dependency` ([#513](https://github.com/rapidsai/kvikio/pull/513)) [@jakirkham](https://github.com/jakirkham)
- S3 benchmark: adding cudf-kvikio and cudf-fsspec ([#509](https://github.com/rapidsai/kvikio/pull/509)) [@madsbk](https://github.com/madsbk)
- Use Cython&#39;s `array` to back `Py_ssize_t[::1]` ([#504](https://github.com/rapidsai/kvikio/pull/504)) [@jakirkham](https://github.com/jakirkham)
- Mark all of `Array`&#39;s `nogil` `cdef` functions as `noexcept` ([#502](https://github.com/rapidsai/kvikio/pull/502)) [@jakirkham](https://github.com/jakirkham)
- Simplify `_to_string` encoding of Python `str`s ([#498](https://github.com/rapidsai/kvikio/pull/498)) [@jakirkham](https://github.com/jakirkham)
- make conda installs in CI stricter ([#495](https://github.com/rapidsai/kvikio/pull/495)) [@jameslamb](https://github.com/jameslamb)
- Small improvements ([#493](https://github.com/rapidsai/kvikio/pull/493)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Prune workflows based on changed files ([#489](https://github.com/rapidsai/kvikio/pull/489)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remote IO: S3 support ([#479](https://github.com/rapidsai/kvikio/pull/479)) [@madsbk](https://github.com/madsbk)
- Use nvcomp wheel instead of bundling nvcomp ([#478](https://github.com/rapidsai/kvikio/pull/478)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Touch ups to `Array` ([#475](https://github.com/rapidsai/kvikio/pull/475)) [@jakirkham](https://github.com/jakirkham)
- Remote IO: http support ([#464](https://github.com/rapidsai/kvikio/pull/464)) [@madsbk](https://github.com/madsbk)

# kvikio 24.10.00 (9 Oct 2024)

## 🚨 Breaking Changes

- Remove now unnecessary variable ([#466](https://github.com/rapidsai/kvikio/pull/466)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Fix no-cuda build ([#484](https://github.com/rapidsai/kvikio/pull/484)) [@madsbk](https://github.com/madsbk)
- Add license to kvikio wheel ([#481](https://github.com/rapidsai/kvikio/pull/481)) [@raydouglass](https://github.com/raydouglass)
- Fix release-after-main bug in `get_primary_cuda_context` ([#472](https://github.com/rapidsai/kvikio/pull/472)) [@madsbk](https://github.com/madsbk)
- Intentionally leak static CUDA resources to avoid crash (part 2) ([#462](https://github.com/rapidsai/kvikio/pull/462)) [@kingcrimsontianyu](https://github.com/kingcrimsontianyu)
- Fix several issues with wheel builds ([#439](https://github.com/rapidsai/kvikio/pull/439)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Fix missing 24.08 =&gt; 24.10 renames ([#423](https://github.com/rapidsai/kvikio/pull/423)) [@madsbk](https://github.com/madsbk)

## 📖 Documentation

- Docs cleanup ([#480](https://github.com/rapidsai/kvikio/pull/480)) [@madsbk](https://github.com/madsbk)

## 🚀 New Features

- Update nvcomp to 4.0.1 ([#449](https://github.com/rapidsai/kvikio/pull/449)) [@vuule](https://github.com/vuule)

## 🛠️ Improvements

- nvtx support macos ([#474](https://github.com/rapidsai/kvikio/pull/474)) [@madsbk](https://github.com/madsbk)
- Update update-version.sh to use packaging lib ([#469](https://github.com/rapidsai/kvikio/pull/469)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove now unnecessary variable ([#466](https://github.com/rapidsai/kvikio/pull/466)) [@vyasr](https://github.com/vyasr)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#465](https://github.com/rapidsai/kvikio/pull/465)) [@jameslamb](https://github.com/jameslamb)
- Update to flake8 7.1.1. ([#463](https://github.com/rapidsai/kvikio/pull/463)) [@bdice](https://github.com/bdice)
- remove pytest dependency in kvikio conda recipe ([#461](https://github.com/rapidsai/kvikio/pull/461)) [@jameslamb](https://github.com/jameslamb)
- Add support for Python 3.12, relax numcodecs pin ([#458](https://github.com/rapidsai/kvikio/pull/458)) [@jameslamb](https://github.com/jameslamb)
- Skip cufile and nvtx3 when cuda isn&#39;t found ([#455](https://github.com/rapidsai/kvikio/pull/455)) [@madsbk](https://github.com/madsbk)
- Non-static bounce buffer option ([#454](https://github.com/rapidsai/kvikio/pull/454)) [@madsbk](https://github.com/madsbk)
- Use nvcomp defaults for algo options. ([#450](https://github.com/rapidsai/kvikio/pull/450)) [@bdice](https://github.com/bdice)
- Update rapidsai/pre-commit-hooks ([#448](https://github.com/rapidsai/kvikio/pull/448)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Make the bounce buffer size configurable ([#447](https://github.com/rapidsai/kvikio/pull/447)) [@madsbk](https://github.com/madsbk)
- Update `build.sh` ([#445](https://github.com/rapidsai/kvikio/pull/445)) [@madsbk](https://github.com/madsbk)
- Drop Python 3.9 support ([#443](https://github.com/rapidsai/kvikio/pull/443)) [@jameslamb](https://github.com/jameslamb)
- Remove NumPy &lt;2 pin ([#441](https://github.com/rapidsai/kvikio/pull/441)) [@seberg](https://github.com/seberg)
- C++ Test ([#440](https://github.com/rapidsai/kvikio/pull/440)) [@madsbk](https://github.com/madsbk)
- `config_path()`: return a reference ([#438](https://github.com/rapidsai/kvikio/pull/438)) [@madsbk](https://github.com/madsbk)
- Stream sync before calling cufile ([#437](https://github.com/rapidsai/kvikio/pull/437)) [@madsbk](https://github.com/madsbk)
- Speeding up Python tests ([#435](https://github.com/rapidsai/kvikio/pull/435)) [@madsbk](https://github.com/madsbk)
- Cython refactoring ([#434](https://github.com/rapidsai/kvikio/pull/434)) [@madsbk](https://github.com/madsbk)
- Update pre-commit hooks ([#433](https://github.com/rapidsai/kvikio/pull/433)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#432](https://github.com/rapidsai/kvikio/pull/432)) [@bdice](https://github.com/bdice)
- Use tool.scikit-build.cmake.version, set scikit-build-core minimum-version ([#430](https://github.com/rapidsai/kvikio/pull/430)) [@jameslamb](https://github.com/jameslamb)
- build.sh: --pydevelop ([#429](https://github.com/rapidsai/kvikio/pull/429)) [@madsbk](https://github.com/madsbk)

# kvikio 24.08.00 (7 Aug 2024)

## 🐛 Bug Fixes

- fix compat_mode check typo ([#421](https://github.com/rapidsai/kvikio/pull/421)) [@madsbk](https://github.com/madsbk)
- Fix `O_DIRECT` in `defined` ([#416](https://github.com/rapidsai/kvikio/pull/416)) [@jakirkham](https://github.com/jakirkham)
- cufile-wrapper: include the cuda-wrapper ([#413](https://github.com/rapidsai/kvikio/pull/413)) [@madsbk](https://github.com/madsbk)
- Rename `.devcontainer`s for CUDA 12.5 ([#406](https://github.com/rapidsai/kvikio/pull/406)) [@jakirkham](https://github.com/jakirkham)
- Use `#ifdef` to guard `O_DIRECT` usage ([#404](https://github.com/rapidsai/kvikio/pull/404)) [@jakirkham](https://github.com/jakirkham)
- Properly quote items in __all__ ([#399](https://github.com/rapidsai/kvikio/pull/399)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 🛠️ Improvements

- Benchmarks clean up ([#424](https://github.com/rapidsai/kvikio/pull/424)) [@madsbk](https://github.com/madsbk)
- split up CUDA-suffixed dependencies in dependencies.yaml ([#411](https://github.com/rapidsai/kvikio/pull/411)) [@jameslamb](https://github.com/jameslamb)
- Don&#39;t open with `O_DIRECT` when in compat mode ([#410](https://github.com/rapidsai/kvikio/pull/410)) [@madsbk](https://github.com/madsbk)
- Fetch and use BS thread_pool ([#408](https://github.com/rapidsai/kvikio/pull/408)) [@madsbk](https://github.com/madsbk)
- Use workflow branch 24.08 again ([#407](https://github.com/rapidsai/kvikio/pull/407)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Enable cuFile on ARM ([#405](https://github.com/rapidsai/kvikio/pull/405)) [@jakirkham](https://github.com/jakirkham)
- Build and test with CUDA 12.5.1 ([#401](https://github.com/rapidsai/kvikio/pull/401)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#400](https://github.com/rapidsai/kvikio/pull/400)) [@jameslamb](https://github.com/jameslamb)
- Use verify-alpha-spec hook ([#398](https://github.com/rapidsai/kvikio/pull/398)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Avoid --find-links in wheel jobs ([#397](https://github.com/rapidsai/kvikio/pull/397)) [@jameslamb](https://github.com/jameslamb)
- Adding NVTX annotations ([#395](https://github.com/rapidsai/kvikio/pull/395)) [@madsbk](https://github.com/madsbk)
- Clean up error handling ([#394](https://github.com/rapidsai/kvikio/pull/394)) [@madsbk](https://github.com/madsbk)
- Move content of the Python pkg to subdirectory python/kvikio/ ([#392](https://github.com/rapidsai/kvikio/pull/392)) [@hcho3](https://github.com/hcho3)
- Adopt CI/packaging codeowners ([#391](https://github.com/rapidsai/kvikio/pull/391)) [@bdice](https://github.com/bdice)
- Remove text builds of documentation ([#390](https://github.com/rapidsai/kvikio/pull/390)) [@vyasr](https://github.com/vyasr)
- use rapids-build-backend ([#389](https://github.com/rapidsai/kvikio/pull/389)) [@jameslamb](https://github.com/jameslamb)
- remove unnecessary &#39;setuptools&#39; dependency ([#385](https://github.com/rapidsai/kvikio/pull/385)) [@jameslamb](https://github.com/jameslamb)
- add --rm and --name to devcontainer run args ([#370](https://github.com/rapidsai/kvikio/pull/370)) [@trxcllnt](https://github.com/trxcllnt)
- add wheel output ([#369](https://github.com/rapidsai/kvikio/pull/369)) [@msarahan](https://github.com/msarahan)

# kvikio 24.06.00 (5 Jun 2024)

## 🐛 Bug Fixes

- Fix conda recipe for kvikio ([#381](https://github.com/rapidsai/kvikio/pull/381)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Move cuFile linking to kvikio target ([#379](https://github.com/rapidsai/kvikio/pull/379)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 🛠️ Improvements

- Initial Python Interface for cufile Async IO ([#376](https://github.com/rapidsai/kvikio/pull/376)) [@K-Wu](https://github.com/K-Wu)
- Enable Warnings as errors for Python tests ([#375](https://github.com/rapidsai/kvikio/pull/375)) [@mroeschke](https://github.com/mroeschke)
- POSIX IO: use async memcpy ([#374](https://github.com/rapidsai/kvikio/pull/374)) [@madsbk](https://github.com/madsbk)
- No-CUDA build ([#373](https://github.com/rapidsai/kvikio/pull/373)) [@madsbk](https://github.com/madsbk)
- Refactor: move cufile header checks to the shim layer ([#372](https://github.com/rapidsai/kvikio/pull/372)) [@madsbk](https://github.com/madsbk)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#371](https://github.com/rapidsai/kvikio/pull/371)) [@hcho3](https://github.com/hcho3)
- Add deflate codec ([#364](https://github.com/rapidsai/kvikio/pull/364)) [@ap--](https://github.com/ap--)

# kvikio 24.04.00 (10 Apr 2024)

## 🚨 Breaking Changes

- remove legate ([#345](https://github.com/rapidsai/kvikio/pull/345)) [@madsbk](https://github.com/madsbk)

## 🐛 Bug Fixes

- Update pre-commit-hooks to v0.0.3 ([#354](https://github.com/rapidsai/kvikio/pull/354)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Defer cuFile feature checks until finding kvikio package ([#342](https://github.com/rapidsai/kvikio/pull/342)) [@bdice](https://github.com/bdice)
- handle more RAPIDS version formats in update-version.sh ([#331](https://github.com/rapidsai/kvikio/pull/331)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Support CUDA 12.2 ([#328](https://github.com/rapidsai/kvikio/pull/328)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- doc: cuda toolkit 12.0 =&gt; 12.2 ([#360](https://github.com/rapidsai/kvikio/pull/360)) [@madsbk](https://github.com/madsbk)
- Use `conda env create --yes` instead of `--force` ([#357](https://github.com/rapidsai/kvikio/pull/357)) [@bdice](https://github.com/bdice)
- Prevent path conflict in builds ([#356](https://github.com/rapidsai/kvikio/pull/356)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add upper bound to prevent usage of NumPy 2 ([#353](https://github.com/rapidsai/kvikio/pull/353)) [@bdice](https://github.com/bdice)
- Replace local copyright check with pre-commit-hooks verify-copyright ([#352](https://github.com/rapidsai/kvikio/pull/352)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove hard-coding of RAPIDS version where possible ([#351](https://github.com/rapidsai/kvikio/pull/351)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add support for Python 3.11, require NumPy 1.23+ ([#350](https://github.com/rapidsai/kvikio/pull/350)) [@jameslamb](https://github.com/jameslamb)
- remove legate ([#345](https://github.com/rapidsai/kvikio/pull/345)) [@madsbk](https://github.com/madsbk)
- Update devcontainers to CUDA Toolkit 12.2 ([#344](https://github.com/rapidsai/kvikio/pull/344)) [@trxcllnt](https://github.com/trxcllnt)
- target branch-24.04 for GitHub Actions workflows ([#343](https://github.com/rapidsai/kvikio/pull/343)) [@jameslamb](https://github.com/jameslamb)
- Add environment-agnostic scripts for running ctests and pytests ([#338](https://github.com/rapidsai/kvikio/pull/338)) [@trxcllnt](https://github.com/trxcllnt)
- Update ops-bot.yaml ([#335](https://github.com/rapidsai/kvikio/pull/335)) [@AyodeAwe](https://github.com/AyodeAwe)

# kvikio 24.02.00 (12 Feb 2024)

## 🚨 Breaking Changes

- Switch to scikit-build-core ([#325](https://github.com/rapidsai/kvikio/pull/325)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Exclude tests from builds ([#336](https://github.com/rapidsai/kvikio/pull/336)) [@vyasr](https://github.com/vyasr)
- Update build.sh ([#332](https://github.com/rapidsai/kvikio/pull/332)) [@madsbk](https://github.com/madsbk)

## 🛠️ Improvements

- Remove usages of rapids-env-update ([#329](https://github.com/rapidsai/kvikio/pull/329)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor CUDA versions in dependencies.yaml ([#327](https://github.com/rapidsai/kvikio/pull/327)) [@jameslamb](https://github.com/jameslamb)
- Switch to scikit-build-core ([#325](https://github.com/rapidsai/kvikio/pull/325)) [@vyasr](https://github.com/vyasr)
- Update nvcomp ([#324](https://github.com/rapidsai/kvikio/pull/324)) [@vyasr](https://github.com/vyasr)
- Add timer to basic_io example ([#321](https://github.com/rapidsai/kvikio/pull/321)) [@yncxcw](https://github.com/yncxcw)
- Forward-merge branch-23.12 to branch-24.02 ([#318](https://github.com/rapidsai/kvikio/pull/318)) [@bdice](https://github.com/bdice)
- Re-enable devcontainer CI. ([#285](https://github.com/rapidsai/kvikio/pull/285)) [@trxcllnt](https://github.com/trxcllnt)

# kvikio 23.12.00 (6 Dec 2023)

## 🚨 Breaking Changes

- Update nvcomp to 3.0.4 (includes API changes) ([#314](https://github.com/rapidsai/kvikio/pull/314)) [@vuule](https://github.com/vuule)

## 🐛 Bug Fixes

- Remove duplicated thread-pool API ([#308](https://github.com/rapidsai/kvikio/pull/308)) [@madsbk](https://github.com/madsbk)
- updated the nvcomp notebook to use the new API ([#294](https://github.com/rapidsai/kvikio/pull/294)) [@madsbk](https://github.com/madsbk)

## 🚀 New Features

- Update nvcomp to 3.0.4 (includes API changes) ([#314](https://github.com/rapidsai/kvikio/pull/314)) [@vuule](https://github.com/vuule)

## 🛠️ Improvements

- Build concurrency for nightly and merge triggers ([#319](https://github.com/rapidsai/kvikio/pull/319)) [@bdice](https://github.com/bdice)
- Revert rapids-cmake branch. ([#316](https://github.com/rapidsai/kvikio/pull/316)) [@bdice](https://github.com/bdice)
- Support no compressor in `open_cupy_array()` ([#312](https://github.com/rapidsai/kvikio/pull/312)) [@madsbk](https://github.com/madsbk)
- Update `shared-action-workflows` references ([#305](https://github.com/rapidsai/kvikio/pull/305)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use branch-23.12 workflows. ([#304](https://github.com/rapidsai/kvikio/pull/304)) [@bdice](https://github.com/bdice)
- Update rapids-cmake functions to non-deprecated signatures ([#301](https://github.com/rapidsai/kvikio/pull/301)) [@robertmaynard](https://github.com/robertmaynard)
- Unify the CUDA Codecs ([#298](https://github.com/rapidsai/kvikio/pull/298)) [@madsbk](https://github.com/madsbk)
- Improve performance of nvCOMP batch codec. ([#293](https://github.com/rapidsai/kvikio/pull/293)) [@Alexey-Kamenev](https://github.com/Alexey-Kamenev)
- Merge branch-23.10 into branch-23.12 and fix devcontainer CI workflow. ([#292](https://github.com/rapidsai/kvikio/pull/292)) [@bdice](https://github.com/bdice)
- kvikio: Build CUDA 12.0 ARM conda packages. ([#282](https://github.com/rapidsai/kvikio/pull/282)) [@bdice](https://github.com/bdice)

# kvikio 23.10.00 (11 Oct 2023)

## 🚨 Breaking Changes

- Update to Cython 3.0.0 ([#258](https://github.com/rapidsai/kvikio/pull/258)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Add numcodecs pin ([#300](https://github.com/rapidsai/kvikio/pull/300)) [@vyasr](https://github.com/vyasr)
- Add missed filename to sed_runner call ([#286](https://github.com/rapidsai/kvikio/pull/286)) [@raydouglass](https://github.com/raydouglass)
- Use `conda mambabuild` not `mamba mambabuild` ([#278](https://github.com/rapidsai/kvikio/pull/278)) [@bdice](https://github.com/bdice)
- fixes #254 ([#262](https://github.com/rapidsai/kvikio/pull/262)) [@madsbk](https://github.com/madsbk)

## 📖 Documentation

- minor doc fixes ([#279](https://github.com/rapidsai/kvikio/pull/279)) [@madsbk](https://github.com/madsbk)
- Docs ([#268](https://github.com/rapidsai/kvikio/pull/268)) [@madsbk](https://github.com/madsbk)
- Zarr notebook ([#261](https://github.com/rapidsai/kvikio/pull/261)) [@madsbk](https://github.com/madsbk)

## 🛠️ Improvements

- Use branch-23.10 for devcontainers workflow. ([#289](https://github.com/rapidsai/kvikio/pull/289)) [@bdice](https://github.com/bdice)
- Update image names ([#284](https://github.com/rapidsai/kvikio/pull/284)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update to clang 16.0.6. ([#280](https://github.com/rapidsai/kvikio/pull/280)) [@bdice](https://github.com/bdice)
- Update doxygen to 1.9.1 ([#277](https://github.com/rapidsai/kvikio/pull/277)) [@vyasr](https://github.com/vyasr)
- Async I/O using by-value arguments ([#275](https://github.com/rapidsai/kvikio/pull/275)) [@madsbk](https://github.com/madsbk)
- Zarr-IO Benchmark ([#274](https://github.com/rapidsai/kvikio/pull/274)) [@madsbk](https://github.com/madsbk)
- Add KvikIO devcontainers ([#273](https://github.com/rapidsai/kvikio/pull/273)) [@trxcllnt](https://github.com/trxcllnt)
- async: fall back to blocking ([#272](https://github.com/rapidsai/kvikio/pull/272)) [@madsbk](https://github.com/madsbk)
- Unify batch and stream API check ([#271](https://github.com/rapidsai/kvikio/pull/271)) [@madsbk](https://github.com/madsbk)
- Use `copy-pr-bot` ([#269](https://github.com/rapidsai/kvikio/pull/269)) [@ajschmidt8](https://github.com/ajschmidt8)
- Zarr+CuPy+GDS+nvCOMP made easy ([#267](https://github.com/rapidsai/kvikio/pull/267)) [@madsbk](https://github.com/madsbk)
- Remove sphinx pinning ([#260](https://github.com/rapidsai/kvikio/pull/260)) [@vyasr](https://github.com/vyasr)
- Initial changes to support cufile stream I/O. ([#259](https://github.com/rapidsai/kvikio/pull/259)) [@tell-rebanta](https://github.com/tell-rebanta)
- Update to Cython 3.0.0 ([#258](https://github.com/rapidsai/kvikio/pull/258)) [@vyasr](https://github.com/vyasr)
- Modernize Python build ([#257](https://github.com/rapidsai/kvikio/pull/257)) [@vyasr](https://github.com/vyasr)
- Enable roundtrip for nvCOMP batch codecs. ([#253](https://github.com/rapidsai/kvikio/pull/253)) [@Alexey-Kamenev](https://github.com/Alexey-Kamenev)

# kvikio 23.08.00 (9 Aug 2023)

## 🐛 Bug Fixes

- Add nvcomp support to older CUDA 11 versions on aarch64. ([#255](https://github.com/rapidsai/kvikio/pull/255)) [@bdice](https://github.com/bdice)
- Unify `KVIKIO_CUFILE_FOUND` ([#243](https://github.com/rapidsai/kvikio/pull/243)) [@madsbk](https://github.com/madsbk)
- Disable the batch API when in compatibility mode ([#239](https://github.com/rapidsai/kvikio/pull/239)) [@madsbk](https://github.com/madsbk)
- Fix libcufile dependency. ([#237](https://github.com/rapidsai/kvikio/pull/237)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- KvikIO: Build CUDA 12 packages ([#224](https://github.com/rapidsai/kvikio/pull/224)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Revert CUDA 12.0 CI workflows to branch-23.08. ([#252](https://github.com/rapidsai/kvikio/pull/252)) [@bdice](https://github.com/bdice)
- Add support for nvCOMP batch API ([#249](https://github.com/rapidsai/kvikio/pull/249)) [@Alexey-Kamenev](https://github.com/Alexey-Kamenev)
- Use cuda-version to constrain cudatoolkit. ([#247](https://github.com/rapidsai/kvikio/pull/247)) [@bdice](https://github.com/bdice)
- Make C++ &amp; Python teams owners of `legate/` ([#246](https://github.com/rapidsai/kvikio/pull/246)) [@jakirkham](https://github.com/jakirkham)
- Use nvcomp conda package. ([#245](https://github.com/rapidsai/kvikio/pull/245)) [@bdice](https://github.com/bdice)
- Adding code owners ([#244](https://github.com/rapidsai/kvikio/pull/244)) [@madsbk](https://github.com/madsbk)
- Clean up dependency lists ([#241](https://github.com/rapidsai/kvikio/pull/241)) [@vyasr](https://github.com/vyasr)
- Clean up isort configs ([#240](https://github.com/rapidsai/kvikio/pull/240)) [@vyasr](https://github.com/vyasr)
- Update to CMake 3.26.4 ([#238](https://github.com/rapidsai/kvikio/pull/238)) [@vyasr](https://github.com/vyasr)
- use rapids-upload-docs script ([#234](https://github.com/rapidsai/kvikio/pull/234)) [@AyodeAwe](https://github.com/AyodeAwe)
- Migrate as much as possible to pyproject.toml, stop using versioneer to manage versions, update dependencies.yaml. ([#232](https://github.com/rapidsai/kvikio/pull/232)) [@bdice](https://github.com/bdice)
- Remove documentation build scripts for Jenkins ([#230](https://github.com/rapidsai/kvikio/pull/230)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use the Zarr&#39;s new `getitems()` API ([#131](https://github.com/rapidsai/kvikio/pull/131)) [@madsbk](https://github.com/madsbk)

# kvikio 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Drop Python 3.8 and run Python 3.9 tests/builds ([#206](https://github.com/rapidsai/kvikio/pull/206)) [@shwina](https://github.com/shwina)
- Use the new registry and mapper API of Legate ([#202](https://github.com/rapidsai/kvikio/pull/202)) [@madsbk](https://github.com/madsbk)

## 🐛 Bug Fixes

- Add sccache s3 controls ([#226](https://github.com/rapidsai/kvikio/pull/226)) [@robertmaynard](https://github.com/robertmaynard)
- fixed import of `ArrayLike` and `DTypeLike` ([#219](https://github.com/rapidsai/kvikio/pull/219)) [@madsbk](https://github.com/madsbk)
- `load_library()`: fixed the mode argument, which was ignored by mistake ([#199](https://github.com/rapidsai/kvikio/pull/199)) [@madsbk](https://github.com/madsbk)

## 📖 Documentation

- Update docs ([#218](https://github.com/rapidsai/kvikio/pull/218)) [@madsbk](https://github.com/madsbk)
- Update README. ([#207](https://github.com/rapidsai/kvikio/pull/207)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- Legate HDF5 using kerchunk ([#222](https://github.com/rapidsai/kvikio/pull/222)) [@madsbk](https://github.com/madsbk)
- run docs nightly too ([#221](https://github.com/rapidsai/kvikio/pull/221)) [@AyodeAwe](https://github.com/AyodeAwe)
- C++ bindings to the Batch API ([#220](https://github.com/rapidsai/kvikio/pull/220)) [@madsbk](https://github.com/madsbk)
- mypy: bump to v1.3.0 ([#214](https://github.com/rapidsai/kvikio/pull/214)) [@madsbk](https://github.com/madsbk)
- Update cupy dependency ([#213](https://github.com/rapidsai/kvikio/pull/213)) [@vyasr](https://github.com/vyasr)
- Enable sccache hits for local builds ([#210](https://github.com/rapidsai/kvikio/pull/210)) [@AyodeAwe](https://github.com/AyodeAwe)
- Revert to branch-23.06 for shared-action-workflows ([#209](https://github.com/rapidsai/kvikio/pull/209)) [@shwina](https://github.com/shwina)
- Zarr+nvCOMP ([#208](https://github.com/rapidsai/kvikio/pull/208)) [@madsbk](https://github.com/madsbk)
- Drop Python 3.8 and run Python 3.9 tests/builds ([#206](https://github.com/rapidsai/kvikio/pull/206)) [@shwina](https://github.com/shwina)
- isort clean up ([#205](https://github.com/rapidsai/kvikio/pull/205)) [@madsbk](https://github.com/madsbk)
- Only look for libcufile.so.0 ([#203](https://github.com/rapidsai/kvikio/pull/203)) [@wence-](https://github.com/wence-)
- Use the new registry and mapper API of Legate ([#202](https://github.com/rapidsai/kvikio/pull/202)) [@madsbk](https://github.com/madsbk)
- Remove usage of rapids-get-rapids-version-from-git ([#201](https://github.com/rapidsai/kvikio/pull/201)) [@jjacobelli](https://github.com/jjacobelli)
- Legate Zarr ([#198](https://github.com/rapidsai/kvikio/pull/198)) [@madsbk](https://github.com/madsbk)
- Add API to get compatibility mode status in a FileHandle object ([#197](https://github.com/rapidsai/kvikio/pull/197)) [@vuule](https://github.com/vuule)
- Update clang-format to 16.0.1. ([#196](https://github.com/rapidsai/kvikio/pull/196)) [@bdice](https://github.com/bdice)
- Use ARC V2 self-hosted runners for GPU jobs ([#195](https://github.com/rapidsai/kvikio/pull/195)) [@jjacobelli](https://github.com/jjacobelli)
- Optimize small reads and writes ([#190](https://github.com/rapidsai/kvikio/pull/190)) [@madsbk](https://github.com/madsbk)
- Remove underscore in build string. ([#188](https://github.com/rapidsai/kvikio/pull/188)) [@bdice](https://github.com/bdice)

# kvikio 23.04.00 (6 Apr 2023)

## 🐛 Bug Fixes

- Fallback to use the CUDA primary context ([#189](https://github.com/rapidsai/kvikio/pull/189)) [@madsbk](https://github.com/madsbk)
- posix_io: fix error message and allow `nbytes == 0` on write ([#184](https://github.com/rapidsai/kvikio/pull/184)) [@madsbk](https://github.com/madsbk)
- Support of stream ordered device memory allocations (async mallocs) ([#181](https://github.com/rapidsai/kvikio/pull/181)) [@madsbk](https://github.com/madsbk)

## 🛠️ Improvements

- Implement `build.sh` ([#185](https://github.com/rapidsai/kvikio/pull/185)) [@madsbk](https://github.com/madsbk)
- Legate Support ([#183](https://github.com/rapidsai/kvikio/pull/183)) [@madsbk](https://github.com/madsbk)
- Fix docs build to be `pydata-sphinx-theme=0.13.0` compatible ([#180](https://github.com/rapidsai/kvikio/pull/180)) [@galipremsagar](https://github.com/galipremsagar)
- Update to GCC 11 ([#179](https://github.com/rapidsai/kvikio/pull/179)) [@bdice](https://github.com/bdice)
- Fix GHA build workflow ([#177](https://github.com/rapidsai/kvikio/pull/177)) [@AjayThorve](https://github.com/AjayThorve)
- Remove Jenkins/`gpuCI` references ([#174](https://github.com/rapidsai/kvikio/pull/174)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `ops-bot.yaml`  config ([#173](https://github.com/rapidsai/kvikio/pull/173)) [@ajschmidt8](https://github.com/ajschmidt8)
- nvcomp xfail compression ratios ([#167](https://github.com/rapidsai/kvikio/pull/167)) [@madsbk](https://github.com/madsbk)
- Move date to build string in `conda` recipe ([#165](https://github.com/rapidsai/kvikio/pull/165)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add gh actions ([#164](https://github.com/rapidsai/kvikio/pull/164)) [@AjayThorve](https://github.com/AjayThorve)

# kvikio 23.02.00 (9 Feb 2023)

## 🐛 Bug Fixes

- Manually open and close the cuFile driver ([#160](https://github.com/rapidsai/kvikio/pull/160)) [@madsbk](https://github.com/madsbk)
- nvcomp tests: update hardcoded lengths ([#156](https://github.com/rapidsai/kvikio/pull/156)) [@madsbk](https://github.com/madsbk)

## 🛠️ Improvements

- Update changelog for releases 22.06-22.12. ([#157](https://github.com/rapidsai/kvikio/pull/157)) [@bdice](https://github.com/bdice)
- Add nvcomp to kvikio-exports export set ([#155](https://github.com/rapidsai/kvikio/pull/155)) [@trxcllnt](https://github.com/trxcllnt)
- Use pre-commit for style checks. ([#154](https://github.com/rapidsai/kvikio/pull/154)) [@bdice](https://github.com/bdice)
- Enable copy_prs. ([#151](https://github.com/rapidsai/kvikio/pull/151)) [@bdice](https://github.com/bdice)

# kvikio 22.12.00 (4 Jan 2023)

## 🐛 Bug Fixes

- Don't use CMake 3.25.0 as it has a FindCUDAToolkit show stopping bug (#146) @robertmaynard
- dlopen: now trying "libcufile.so.1", "libcufile.so.0", "libcufile.so" (#141) @madsbk
- Update nvcomp's expected sizes when testing (#134) @madsbk
- `is_host_memory()`: returns `true` when `CUDA_ERROR_NOT_INITIALIZED` (#133) @madsbk

## 📖 Documentation

- Use rapidsai CODE_OF_CONDUCT.md (#144) @bdice

## 🛠️ Improvements

- Dependency clean up (#137) @madsbk
- Overload `numpy.fromfile()` and `cupy.fromfile()` (#135) @madsbk

# kvikio 22.10.00 (25 Oct 2022)

## 🚨 Breaking Changes

- Rename `reset_num_threads()` and `reset_task_size()` (#123) @madsbk

## 🐛 Bug Fixes

- Fixing `kvikio_dev_cuda11.5.yml` and clean up (#122) @madsbk

## 📖 Documentation

- Document that minimum required CMake version is now 3.23.1 (#132) @robertmaynard

## 🛠️ Improvements

- Use Zarr v2.13.0a2 (#129) @madsbk
- Allow cupy 11 (#128) @galipremsagar
- Set version when dlopen() cuda and cufile (#127) @madsbk
- Fall back to compat mode if we cannot open the file with `O_DIRECT` (#126) @madsbk
- Update versioneer to v0.22 (#124) @madsbk
- Rename `reset_num_threads()` and `reset_task_size()` (#123) @madsbk
- document channel_priority for conda (#121) @dcherian
- Update nvComp bindings to 2.3.3 (#120) @thomcom
- Use rapids-cmake 22.10 best practice for RAPIDS.cmake location (#118) @robertmaynard
- Standalone Downstream C++ Build Example (#29) @madsbk

# kvikio 22.08.00 (18 Aug 2022)

## 🚨 Breaking Changes

- Fix typo in GDS availability check (#78) @jakirkham

## 🐛 Bug Fixes

- CI: install cuDF to test nvCOMP (#108) @madsbk
- Require `python` in `run` (#101) @jakirkham
- Check stub error (#86) @madsbk
- FindcuFile now searches in the current CUDA Toolkit location (#85) @robertmaynard
- Fix typo in GDS availability check (#78) @jakirkham

## 📖 Documentation

- Defer loading of `custom.js` (#114) @galipremsagar
- Add stable channel install instruction (#107) @dcherian
- Use documented header template for `doxygen` (#105) @galipremsagar
- Fix issues with day & night modes in docs (#95) @galipremsagar
- add Notes related to page-aligned read/write methods to the Python docstrings (#91) @grlee77
- minor docstring fixes (#89) @grlee77

## 🛠️ Improvements

- Conda environment file (#103) @madsbk
- CI: build without cuFile (#100) @madsbk
- Support host memory (#82) @madsbk

# kvikio 22.06.00 (7 Jun 2022)

## 🐛 Bug Fixes

- Mark detail functions as inline (#69) @vyasr
- Embed Cython docstrings in generated files. (#58) @vyasr
- Add new files to update version script (#53) @charlesbluca

## 🛠️ Improvements

- Fix conda recipes (#74) @Ethyling
- Use conda compilers (#71) @Ethyling
- dlopen `libcuda.so` (#70) @madsbk
- Use CMake provided targets for the cuda driver and dl libraries (#68) @robertmaynard
- Use namespace kvikio::detail (#67) @madsbk
- Build kvikio using libkvikio from CPU job in GPU job (#64) @Ethyling
- Use conda to build python packages during GPU tests (#62) @Ethyling
- `convert_size2off()`: fix different signedness (#60) @madsbk
- Keep the rapids-cmake version insync with calver (#59) @robertmaynard
- compat_mode per `FileHandle` (#54) @madsbk
- Add `ops-bot.yaml` config file (#52) @ajschmidt8
- python: bump versions to v22.06 (#51) @madsbk
