# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# The plugin is a standalone executable, and does not link libkvikio.
set(NSYS_PLUGIN_INSTALL_DIR share/kvikio/nsys-plugins/kvikio_nic)

add_executable(kvikio_nic_nsys_plugin kvikio_nic_nsys_plugin.cpp nic_monitor.cpp)

set_target_properties(
  kvikio_nic_nsys_plugin
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             INSTALL_RPATH ""
)

target_link_libraries(kvikio_nic_nsys_plugin PRIVATE nvtx3::nvtx3-cpp ${CMAKE_DL_LIBS})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    kvikio_nic_nsys_plugin PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Werror;-Wno-unknown-pragmas>"
  )
endif()

install(
  TARGETS kvikio_nic_nsys_plugin
  COMPONENT nsys_plugin
  DESTINATION ${NSYS_PLUGIN_INSTALL_DIR}
)
install(
  FILES nsys-plugin.yaml
  COMPONENT nsys_plugin
  DESTINATION ${NSYS_PLUGIN_INSTALL_DIR}
)
