/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

FileWrapper::FileWrapper(std::string const& file_path,
                         std::string const& flags,
                         bool o_direct,
                         mode_t mode)
{
  KVIKIO_NVTX_FUNC_RANGE();
  open(file_path, flags, o_direct, mode);
}

FileWrapper::~FileWrapper() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  close();
}

FileWrapper::FileWrapper(FileWrapper&& o) noexcept : _fd(std::exchange(o._fd, -1)) {}

FileWrapper& FileWrapper::operator=(FileWrapper&& o) noexcept
{
  _fd = std::exchange(o._fd, -1);
  return *this;
}

void FileWrapper::open(std::string const& file_path,
                       std::string const& flags,
                       bool o_direct,
                       mode_t mode)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (!opened()) { _fd = open_fd(file_path, flags, o_direct, mode); }
}

bool FileWrapper::opened() const noexcept { return _fd != -1; }

void FileWrapper::close() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (opened()) {
    if (::close(_fd) != 0) { KVIKIO_LOG_ERROR("File cannot be closed"); }
    _fd = -1;
  }
}

int FileWrapper::fd() const noexcept { return _fd; }

CUFileHandleWrapper::~CUFileHandleWrapper() noexcept { unregister_handle(); }

CUFileHandleWrapper::CUFileHandleWrapper(CUFileHandleWrapper&& o) noexcept
  : _handle{std::exchange(o._handle, {})}, _registered{std::exchange(o._registered, false)}
{
}

CUFileHandleWrapper& CUFileHandleWrapper::operator=(CUFileHandleWrapper&& o) noexcept
{
  _handle     = std::exchange(o._handle, {});
  _registered = std::exchange(o._registered, false);
  return *this;
}

std::optional<CUfileError_t> CUFileHandleWrapper::register_handle(int fd) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::optional<CUfileError_t> error_code;
  if (registered()) { return error_code; }

  // Create a cuFile handle, if not in compatibility mode
  CUfileDescr_t desc{};  // It is important to set to zero!
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
  desc.handle.fd = fd;
  error_code     = cuFileAPI::instance().HandleRegister(&_handle, &desc);
  if (error_code.value().err == CU_FILE_SUCCESS) { _registered = true; }
  return error_code;
}

bool CUFileHandleWrapper::registered() const noexcept { return _registered; }

CUfileHandle_t CUFileHandleWrapper::handle() const noexcept { return _handle; }

void CUFileHandleWrapper::unregister_handle() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (registered()) {
    cuFileAPI::instance().HandleDeregister(_handle);
    _registered = false;
  }
}

int open_fd_parse_flags(std::string const& flags, bool o_direct)
{
  KVIKIO_NVTX_FUNC_RANGE();
  int file_flags = -1;
  KVIKIO_EXPECT(!flags.empty(), "Unknown file open flag", std::invalid_argument);
  switch (flags[0]) {
    case 'r':
      file_flags = O_RDONLY;
      if (flags.length() > 1 && flags[1] == '+') { file_flags = O_RDWR; }
      break;
    case 'w':
      file_flags = O_WRONLY;
      if (flags.length() > 1 && flags[1] == '+') { file_flags = O_RDWR; }
      file_flags |= O_CREAT | O_TRUNC;
      break;
    case 'a': KVIKIO_FAIL("Open flag 'a' isn't supported", std::invalid_argument);
    default: KVIKIO_FAIL("Unknown file open flag", std::invalid_argument);
  }
  file_flags |= O_CLOEXEC;
  if (o_direct) {
#if defined(O_DIRECT)
    file_flags |= O_DIRECT;
#else
    KVIKIO_FAIL("'o_direct' flag unsupported on this platform", std::invalid_argument);
#endif
  }
  return file_flags;
}

int open_fd(std::string const& file_path, std::string const& flags, bool o_direct, mode_t mode)
{
  KVIKIO_NVTX_FUNC_RANGE();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  int fd = ::open(file_path.c_str(), open_fd_parse_flags(flags, o_direct), mode);
  SYSCALL_CHECK(fd, "Unable to open file.");
  return fd;
}

[[nodiscard]] int open_flags(int fd)
{
  KVIKIO_NVTX_FUNC_RANGE();
  int ret = fcntl(fd, F_GETFL);  // NOLINT(cppcoreguidelines-pro-type-vararg)
  SYSCALL_CHECK(ret, "Unable to retrieve open flags.");
  return ret;
}

[[nodiscard]] std::size_t get_file_size(std::string const& file_path)
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::string const flags{"r"};
  bool const o_direct{false};
  mode_t const mode{FileHandle::m644};
  auto fd     = open_fd(file_path, flags, o_direct, mode);
  auto result = get_file_size(fd);
  SYSCALL_CHECK(close(fd));
  return result;
}

[[nodiscard]] std::size_t get_file_size(int file_descriptor)
{
  KVIKIO_NVTX_FUNC_RANGE();
  struct stat st{};
  int ret = fstat(file_descriptor, &st);
  SYSCALL_CHECK(ret, "Unable to query file size.");
  return static_cast<std::size_t>(st.st_size);
}

std::pair<std::size_t, std::size_t> get_page_cache_info(std::string const& file_path)
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::string const flags{"r"};
  bool const o_direct{false};
  mode_t const mode{FileHandle::m644};
  auto fd     = open_fd(file_path, flags, o_direct, mode);
  auto result = get_page_cache_info(fd);
  SYSCALL_CHECK(close(fd));
  return result;
}

std::pair<std::size_t, std::size_t> get_page_cache_info(int fd)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto file_size = get_file_size(fd);

  std::size_t offset{0u};
  auto addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, offset);
  SYSCALL_CHECK(addr, "mmap failed.", MAP_FAILED);

  std::size_t num_pages = (file_size + get_page_size() - 1) / get_page_size();
  std::vector<unsigned char> is_in_page_cache(num_pages, {});
  SYSCALL_CHECK(mincore(addr, file_size, is_in_page_cache.data()));
  std::size_t num_pages_in_page_cache{0u};
  for (std::size_t page_idx = 0; page_idx < is_in_page_cache.size(); ++page_idx) {
    // The least significant bit of each byte will be set if the corresponding page is currently
    // resident in memory, and be clear otherwise. The settings of the other bits in each byte are
    // undefined
    if (static_cast<int>(is_in_page_cache[page_idx]) & 0x1) { ++num_pages_in_page_cache; }
  }

  SYSCALL_CHECK(munmap(addr, file_size));
  return {num_pages_in_page_cache, num_pages};
}

bool clear_page_cache(bool reclaim_dentries_and_inodes, bool clear_dirty_pages)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (clear_dirty_pages) { sync(); }
  std::string param = reclaim_dentries_and_inodes ? "3" : "1";

  auto exec_cmd = [](std::string_view cmd) -> bool {
    // Prevent the output from the command from mixing with the original process' output.
    fflush(nullptr);
    // popen only handles stdout. Switch stderr and stdout to only capture stderr.
    auto const redirected_cmd =
      std::string{"( "}.append(cmd).append(" 3>&2 2>&1 1>&3) 2>/dev/null");
    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(redirected_cmd.c_str(), "r"), pclose);
    KVIKIO_EXPECT(pipe != nullptr, "popen() failed", GenericSystemError);

    std::array<char, 128> buffer;
    std::string error_out;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      error_out += buffer.data();
    }
    return error_out.empty();
  };

  std::array cmds{
    // Special case:
    // - Unprivileged users who cannot execute `/usr/bin/sudo` but can execute `/sbin/sysctl`, and
    // - Superuser
    std::string{"/sbin/sysctl vm.drop_caches=" + param},
    // General case:
    // - Unprivileged users who can execute `sudo`, and
    // - Superuser
    std::string{"sudo /sbin/sysctl vm.drop_caches=" + param}};

  for (auto const& cmd : cmds) {
    if (exec_cmd(cmd)) { return true; }
  }
  return false;
}

BlockDeviceInfo get_block_device_info(std::string const& file_path)
{
  BlockDeviceInfo block_dev_info{};
  unsigned dev_major_id{};
  unsigned dev_minor_id{};

  struct stat st;
  SYSCALL_CHECK(stat(file_path.c_str(), &st));

  dev_major_id = major(st.st_dev);
  dev_minor_id = minor(st.st_dev);

  // Construct sysfs path (which is a symlink) for the block device
  // e.g., /sys/dev/block/259:8
  std::string sysfs_path =
    "/sys/dev/block/" + std::to_string(dev_major_id) + ":" + std::to_string(dev_minor_id);

  // Resolve the symlink to the actual sysfs device path
  // e.g.,
  // - NVMe: /sys/devices/pci0000:00/.../nvme/nvme1/nvme1n1/nvme1n1p3
  // - SATA: /sys/devices/pci0000:00/.../block/sdb/sdb1
  char resolved_path[PATH_MAX];
  SYSCALL_CHECK(realpath(sysfs_path.c_str(), resolved_path),
                "Path failed to resolve",
                static_cast<char*>(nullptr));

  // If this is a partition, walk up to the parent block device.
  // Partitions have a "partition" file in their sysfs directory.
  std::filesystem::path cur_path(resolved_path);
  if (std::filesystem::exists(cur_path / "partition")) { cur_path = cur_path.parent_path(); }

  // After walking up, cur_path points to the base block device
  // e.g.,
  // - NVMe: /sys/devices/pci0000:00/.../nvme/nvme1/nvme1n1
  // - SATA: /sys/devices/pci0000:00/.../block/sdb

  // Extract block device name block_dev_name
  // e.g.,
  // - NVMe: nvme1n1
  // - SATA: sdb
  std::string block_dev_name{cur_path.filename()};

  std::string dev_file;

  // For NVMe, multiple namespaces (nvme0n1, nvme0n2) share the same controller (nvme0) and thus the
  // same physical hardware. We want to create thread pools per controller instead of per namespace.
  static std::regex const nvme_pattern(R"(^nvme\d+n\d+$)");
  if (std::regex_match(block_dev_name, nvme_pattern)) {
    // For NVMe, walk up to controller directory
    // e.g., cur_path:        /sys/devices/pci0000:00/.../nvme/nvme0/nvme0n1
    //       controller_path: /sys/devices/pci0000:00/.../nvme/nvme0
    //       dev_file:        /sys/devices/pci0000:00/.../nvme/nvme0/dev
    auto controller_path = cur_path.parent_path();
    dev_file             = (controller_path / "dev").string();
    block_dev_name       = controller_path.filename().string();  // e.g., "nvme0"
  } else {
    // For non-NVMe devices, use the block device's own dev file
    // e.g., /sys/devices/pci0000:00/.../block/sdb/dev
    dev_file = cur_path.string() + "/dev";
  }

  // Parse "major:minor" from the dev file
  std::ifstream ifs(dev_file);
  KVIKIO_EXPECT(!ifs.fail(), "Block device file cannot be opened: " + dev_file);
  char colon{};
  ifs >> dev_major_id >> colon >> dev_minor_id;
  KVIKIO_EXPECT(colon == ':',
                "Parsing failed. Unexpected character encountered: " + std::string(1, colon));

  block_dev_info.major = dev_major_id;
  block_dev_info.minor = dev_minor_id;
  block_dev_info.id    = makedev(dev_major_id, dev_minor_id);
  block_dev_info.name  = std::move(block_dev_name);

  return block_dev_info;
}

}  // namespace kvikio
