import parallel_io

if __name__ == "__main__":
    config = {"create_file": True,
              "skip_remaining_tests": True}
    tm = parallel_io.TestManager(config)
    tm.perform_test()

    num_threads_range = [1, 2, 4, 8, 16, 32, 64, 96]
    for num_threads in num_threads_range:
        config = {"create_file": False,
                  "drop_file_cache": False,
                  "num_threads": num_threads}
        tm = parallel_io.TestManager(config)
        tm.perform_test()
