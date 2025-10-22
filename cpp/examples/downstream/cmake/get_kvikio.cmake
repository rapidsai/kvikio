# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to fetch KvikIO, which makes `kvikio::kvikio` available for `target_link_libraries`
function(find_and_configure_kvikio MIN_VERSION)

  CPMFindPackage(
    NAME KvikIO
    VERSION ${MIN_VERSION}
            GIT_REPOSITORY
            https://github.com/rapidsai/kvikio.git
    GIT_TAG branch-${MIN_VERSION}
    GIT_SHALLOW
      TRUE
      SOURCE_SUBDIR
      cpp
    OPTIONS "KvikIO_BUILD_EXAMPLES OFF"
  )

endfunction()

find_and_configure_kvikio("25.12")
