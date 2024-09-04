/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package bindings.kvikio.cufile;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import static org.junit.jupiter.api.Assertions.*;

public class BasicReadWriteTest {

    @Test
    public void testReadBackWrite()
    {
        String libraryPath = System.getProperty("java.library.path");
        System.out.println("Java library path: " + libraryPath);
        
        // Allocate CUDA device memory
        int numInts = 4;
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, numInts*Sizeof.INT);

        // Build host arrays
        int[] hostData = new int[numInts];
        int[] hostDataFilled = new int[numInts];
        for (int i = 0; i < numInts; ++i) {
            hostDataFilled[i]=i;
        }

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

        // Copy data back to host and confirm what was written was read back
        JCuda.cudaMemcpy(Pointer.to(hostData), pointer, numInts*Sizeof.INT, cudaMemcpyDeviceToHost);
        assertArrayEquals(hostData,hostDataFilled);
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
