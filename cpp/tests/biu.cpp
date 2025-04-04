#include <cufile.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

#define CHECK_CUFILE(err_code) check_cufile(err_code, __FILE__, __LINE__)
void check_cufile(CUfileError_t err_code, const char* file, int line)
{
  auto cufile_err_code = err_code.err;  // CUfileOpError
  if (cufile_err_code != CU_FILE_SUCCESS) {
    std::stringstream ss;
    ss << "cuFile error at" << file << ":" << line << std::endl;
    throw std::runtime_error(ss.str());
  }
}

#define EXPECT(condition) expect(condition, __FILE__, __LINE__)
inline void expect(bool condition, const char* file, int line)
{
  if (condition) { return; }
  std::stringstream ss;
  ss << "EXPECT failed at " << file << ":" << line << std::endl;
  throw std::runtime_error(ss.str());
}

class TestManager {
 public:
  TestManager()
  {
    load_library();

    int flags{O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT};
    mode_t mode{S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH};
    _fd = open(_file_path.c_str(), flags, mode);
    EXPECT(_fd != -1);

    CUfileDescr_t desc{};
    desc.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = _fd;
    CHECK_CUFILE(_func_handle_register(&_handle, &desc));
  }

  ~TestManager()
  {
    _func_handle_deregister(_handle);
    EXPECT(close(_fd) == 0);
    std::cout << "test done" << std::endl;
  }

 private:
  void load_library()
  {
    dlerror();
    auto* cufile_lib_handle =
      dlopen(_cufile_lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
    get_symbol(_func_handle_register, cufile_lib_handle, "cuFileHandleRegister");
    get_symbol(_func_handle_deregister, cufile_lib_handle, "cuFileHandleDeregister");
  }

  template <typename F>
  void get_symbol(F& func, void* cufile_lib_handle, std::string const& name)
  {
    dlerror();
    func      = reinterpret_cast<std::decay_t<F>>(dlsym(cufile_lib_handle, name.c_str()));
    auto* err = dlerror();
    if (err != nullptr) { throw std::runtime_error(err); }
  }

  int _fd{};
  CUfileHandle_t _handle{};
  std::string _cufile_lib_path{"/usr/local/cuda/targets/sbsa-linux/lib/libcufile.so.0"};
  std::string _file_path{"/mnt/nvme/biu.bin"};

  std::decay_t<decltype(cuFileHandleRegister)> _func_handle_register;
  std::decay_t<decltype(cuFileHandleDeregister)> _func_handle_deregister;
};

int main()
{
  TestManager tm;
  return 0;
}