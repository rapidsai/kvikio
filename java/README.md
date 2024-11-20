# Java KvikIO Bindings

## Summary
These Java KvikIO bindings for GDS currently support only synchronous read and write IO operations using the underlying cuFile API. Support for batch IO and asynchronous operations are not yet supported.

## Dependencies
The Java KvikIO bindings have been developed to work on Linux based systems and require [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to be installed and for [GDS](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html) to be properly enabled. To compile the shared library it is also necessary to have a JDK installed. To run the included example, it is also necessary to install JCuda as it is used to handle memory allocations and the transfer of data between host and GPU memory. JCuda jar files supporting CUDA 12.x can be found here:
[jcuda-12.0.0.jar](https://repo1.maven.org/maven2/org/jcuda/jcuda/12.0.0/jcuda-12.0.0.jar),
[jcuda-natives-12.0.0.jar](https://repo1.maven.org/maven2/org/jcuda/jcuda-natives/12.0.0/jcuda-natives-12.0.0.jar)

For more information on JCuda and potentially more up to date installation instructions or jar files, see here:
[JCuda](http://javagl.de/jcuda.org/), [JCuda Usage](https://github.com/jcuda/jcuda-main/blob/master/USAGE.md), [JCuda Maven Repo](https://mvnrepository.com/artifact/org.jcuda)

## Compilation and examples
An example for how to use the Java KvikIO bindings can be found in `src/test/java/ai/rapids/kvikio/cufile/BasicReadWriteTest.java`

##### Note: This example has a dependency on JCuda so ensure that when running the example the JCuda shared library files are on the JVM library path along with the `libCuFileJNI.so` file.

### Setup a test file target
##### NOTE: the example as written will default to creating a temporary file in your `/tmp` directory. This directory may not be mounted in a compatible manner for use with GDS on your particular system, causing the example to run in compatibility mode. If this is the case, run the following command replacing `/mnt/nvme/` with your mount directory and update `cufile/BasicReadWriteTest.java` to point to the correct file path.

    touch /mnt/nvme/java_test

### Compile the shared library and Java files with Maven
##### Note: This wil also run the example code

    cd kvikio/java/
    mvn clean install

### Rerun example code with Maven

    cd kvikio/java/
    mvn test
