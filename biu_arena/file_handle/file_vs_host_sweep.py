import file_vs_host

if __name__ == "__main__":
    config = {"create_file": True,
              "skip_remaining_tests": True}
    tm = file_vs_host.TestManager(config)
    tm.perform_test()

    # num_threads_range = [1, 2, 4, 8, 16, 32, 64, 96]
    num_threads_range = [8]
    for num_threads in num_threads_range:
        config = {"create_file": False,
                  "skip_mmap": True,
                  "num_threads": num_threads}
        tm = file_vs_host.TestManager(config)
        tm.perform_test()
