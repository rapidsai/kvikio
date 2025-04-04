import kvikio
import kvikio.cufile_driver

file_name = '/mnt/nvme/segfault.txt'

fd = kvikio.CuFile(file_name, "w")
fd.close()
kvikio.cufile_driver.driver_close()
