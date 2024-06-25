Java Bindings

Summary
These Java KvikIO bindings for GDS currently support only synchronous read and write IO operations using the underlying CuFile API. Support for batch IO and asynchronous operations are not yet supported.

Dependencies
The Java KvikIO bindings have been developed to work on Linux based systems and require CUDA to be installed and for GDS to be properly enabled. Instructions for how to install and enable GDS can be found on NVIDIA's website. To compile the shared library it is also necessary to have a JDK installed. To run the included example, it is also necessary to install JCuda as it is used to handle memory allocations and the transfer of data between host and GPU memory. JCuda jar files supporting CUDA 12.x can be found here:
https://repo1.maven.org/maven2/org/jcuda/jcuda/12.0.0/jcuda-12.0.0.jar
https://repo1.maven.org/maven2/org/jcuda/jcuda-natives/12.0.0/jcuda-natives-12.0.0.jar

Compilation
To recompile the .so file for your local system run the following command. Note: Update the command to reflect the directory where you have installed CUDA and your JDK.

/usr/local/cuda/bin/nvcc -shared -o libCuFileJNI.so -I/usr/local/cuda/include/ -I/usr/lib/jvm/java-21-openjdk-amd64/include/ -I/usr/lib/jvm/java-21-openjdk-amd64/include/linux src/main/native/src/CuFileJni.cpp --compiler-options "-fPIC" -lcufile

The resulting .so file must be in your JVM library path. If it is not already placed on your path in can be included when compiling and running your Java code by including an argument like the following:
-Djava.library.path={path/to/your/so/file/}

Examples
An example for how to use the Java KvikIO bindings can be found in src/main/java/bindings/kvikio/example . Note: This example has a dependency on JCuda so ensure that when running the example the JCuda shared library files are on the JVM library path along with the libCuFileJNI.so file.
