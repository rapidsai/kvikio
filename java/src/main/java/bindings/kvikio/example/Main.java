package bindings.kvikio.example;

import bindings.kvikio.cufile.CuFileReadHandle;
import bindings.kvikio.cufile.CuFileWriteHandle;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

public class Main {
    public static void main(String []args)
    {
        // Allocate CUDA device memory
        int numInts = 4;
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, numInts*Sizeof.INT);

        // Build host arrays, print them out
        int hostData[] = new int[numInts];
        int hostDataFilled[] = new int[numInts];
        for (int i = 0; i < numInts; ++i) {
            hostDataFilled[i]=i;
        }
        System.out.println(Arrays.toString(hostData));
        System.out.println(Arrays.toString(hostDataFilled));

        // Obtain pointer value for allocated CUDA device memory
        long pointerAddress = getPointerAddress(pointer);
        
        // Copy filled data array to GPU and write to file
        JCuda.cudaMemcpy(pointer,Pointer.to(hostDataFilled),numInts*Sizeof.INT,cudaMemcpyHostToDevice);
        CuFileWriteHandle fw = new CuFileWriteHandle("/mnt/nvme/java_test");
        fw.write(pointerAddress, numInts*Sizeof.INT,0,0);
        fw.close();

        // Clear data stored in GPU
        JCuda.cudaMemcpy(pointer,Pointer.to(hostData),numInts*Sizeof.INT,cudaMemcpyHostToDevice);

        // Read data back into GPU
        CuFileReadHandle f = new CuFileReadHandle("/mnt/nvme/java_test");
        f.read(pointerAddress,numInts*Sizeof.INT,0,0);
        f.close();

        // Copy data back to host and confirm what was written was read
        JCuda.cudaMemcpy(Pointer.to(hostData), pointer, numInts*Sizeof.INT, cudaMemcpyDeviceToHost);
        System.out.println(Arrays.toString(hostDataFilled));
        System.out.println(Arrays.toString(hostData));
        JCuda.cudaFree(pointer);
    }

    private static long getPointerAddress(Pointer p)
    {
        // WORKAROUND until a method like CUdeviceptr#getAddress exists
        class PointerWithAddress extends Pointer
        {
            PointerWithAddress(Pointer other)
            {
                super(other);
            }
            long getAddress()
            {
                return getNativePointer() + getByteOffset();
            }
        }
        return new PointerWithAddress(p).getAddress();
    }
};
