/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.kvikio.cufile;

import java.io.File;
import java.io.IOException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

import org.junit.jupiter.api.Test;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import static org.junit.jupiter.api.Assertions.*;

public class BasicReadWriteTest {

    @Test
    public void testReadBackWrite() throws IOException {
        // Allocate CUDA device memory
        int numInts = 4;
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, numInts * Sizeof.INT);

        // Build host arrays
        int[] hostData = new int[numInts];
        int[] hostDataFilled = new int[numInts];
        for (int i = 0; i < numInts; ++i) {
            hostDataFilled[i] = i;
        }

        // Obtain pointer value for allocated CUDA device memory
        long pointerAddress = getPointerAddress(pointer);

        // Copy filled data array to GPU and write to file
        JCuda.cudaMemcpy(pointer, Pointer.to(hostDataFilled), numInts * Sizeof.INT, cudaMemcpyHostToDevice);
        File testFile = File.createTempFile("java_test",".tmp");

        CuFileWriteHandle fw = new CuFileWriteHandle(testFile.getAbsolutePath());
        fw.write(pointerAddress, numInts * Sizeof.INT, 0, 0);
        fw.close();

        // Clear data stored in GPU
        JCuda.cudaMemcpy(pointer, Pointer.to(hostData), numInts * Sizeof.INT, cudaMemcpyHostToDevice);

        // Read data back into GPU
        CuFileReadHandle f = new CuFileReadHandle(testFile.getAbsolutePath());
        f.read(pointerAddress, numInts * Sizeof.INT, 0, 0);
        f.close();

        // Copy data back to host and confirm what was written was read back
        JCuda.cudaMemcpy(Pointer.to(hostData), pointer, numInts * Sizeof.INT, cudaMemcpyDeviceToHost);
        assertArrayEquals(hostData, hostDataFilled);
        JCuda.cudaFree(pointer);
    }

    private static long getPointerAddress(Pointer p) {
        // WORKAROUND until a method like CUdeviceptr#getAddress exists
        class PointerWithAddress extends Pointer {
            PointerWithAddress(Pointer other) {
                super(other);
            }

            long getAddress() {
                return getNativePointer() + getByteOffset();
            }
        }
        return new PointerWithAddress(p).getAddress();
    }
}
