# Java KvikIO Bindings

## Summary
These Java KvikIO bindings for GDS currently support only synchronous read and write IO operations using the underlying CuFile API. Support for batch IO and asynchronous operations are not yet supported.

## Dependencies
The Java KvikIO bindings have been developed to work on Linux based systems and require [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to be installed and for [GDS](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html) to be properly enabled. To compile the shared library it is also necessary to have a JDK installed. To run the included example, it is also necessary to install JCuda as it is used to handle memory allocations and the transfer of data between host and GPU memory. JCuda jar files supporting CUDA 12.x can be found here:
[jcuda-12.0.0.jar](https://repo1.maven.org/maven2/org/jcuda/jcuda/12.0.0/jcuda-12.0.0.jar), 
[jcuda-natives-12.0.0.jar](https://repo1.maven.org/maven2/org/jcuda/jcuda-natives/12.0.0/jcuda-natives-12.0.0.jar)

For more information on JCuda and potentially more up to date installation instructions or jar files, see here:
[JCuda](http://javagl.de/jcuda.org/), [JCuda Usage](https://github.com/jcuda/jcuda-main/blob/master/USAGE.md), [JCuda Maven Repo](https://mvnrepository.com/artifact/org.jcuda)

## Compilation
To recompile the .so file for your local system run the following command. Note: Update the command to reflect the directory where you have installed CUDA and your JDK.

    /usr/local/cuda/bin/nvcc -shared -o libCuFileJNI.so -I/usr/local/cuda/include/ -I/usr/lib/jvm/java-21-openjdk-amd64/include/ -I/usr/lib/jvm/java-21-openjdk-amd64/include/linux src/main/native/src/CuFileJni.cpp --compiler-options "-fPIC" -lcufile

The resulting .so file must be in your JVM library path when running upstream Java programs. If it is not already placed on your path in can be included by including an argument like the following:
    
    -Djava.library.path={path/to/your/so/file/}

## Examples
An example for how to use the Java KvikIO bindings can be found in src/main/java/bindings/kvikio/example . Note: This example has a dependency on JCuda so ensure that when running the example the JCuda shared library files are on the JVM library path along with the libCuFileJNI.so file.

### Specific instructions to run the example using Maven

#### Compile the shared library and Java files with Maven

    cd kvikio/java/
    mvn clean install

#### Setup a test file target NOTE: your mount directory may differ from /mnt/nvme, so update this command appropriately as well as example/Main.java to point to the correct file path.

    touch /mnt/nvme/java_test

#### Run example
    
    cd kvikio/java/
    java -cp target/cufile-24.08.0-SNAPSHOT.jar:$HOME/.m2/repository/org/jcuda/jcuda/12.0.0/jcuda-12.0.0.jar:$HOME/.m2/repository/org/jcuda/jcuda-natives/12.0.0/jcuda-natives-12.0.0.jar -Djava.library.path=./target bindings.kvikio.example.Main

### Specific instructions to run the example from a terminal

#### Compile class files

    cd kvikio/java/src/main/java/bindings/kvikio/cufile
    javac *.java

#### Retrieve Jcuda jar files

    cd kvikio/java/
    mkdir lib
    cd lib
    wget https://repo1.maven.org/maven2/org/jcuda/jcuda/12.0.0/jcuda-12.0.0.jar
    wget https://repo1.maven.org/maven2/org/jcuda/jcuda-natives/12.0.0/jcuda-natives-12.0.0.jar

#### Compile shared library

    cd kvikio/java/lib
    /usr/local/cuda/bin/nvcc -shared -o libCuFileJNI.so -I/usr/local/cuda/include/ -I/usr/lib/jvm/java-21-openjdk-amd64/include/ -I/usr/lib/jvm/java-21-openjdk-amd64/include/linux ../src/main/native/src/CuFileJni.cpp --compiler-options "-fPIC" -lcufile

#### Setup a test file target NOTE: your mount directory may differ from /mnt/nvme, so update this command appropriately as well as example/Main.java to point to the correct file path.

    touch /mnt/nvme/java_test

#### Compile example file

    cd kvikio/java/src/main/java 
    javac -cp .:../../../lib/jcuda-12.0.0.jar:../../../lib/jcuda-natives-12.0.0.jar bindings/kvikio/example/Main.java

#### Run example

    cd kvikio/java/src/main/java 
    java -cp .:../../../lib/jcuda-12.0.0.jar:../../../lib/jcuda-natives-12.0.0.jar -Djava.library.path=../../../lib/ bindings.kvikio.example.main

