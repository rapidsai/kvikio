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

package ai.rapids.kvikio.cufile;

/**
 * The {@code CuFileHandle} class is an abstract base class for managing CuFile
 * resource handles in the native library.
 * It represents a common mechanism for interacting with files via CuFile,
 * storing the native pointer to a resource and
 * ensuring that the resource is properly destroyed when no longer needed.
 * <p>
 * This class initializes the CuFile library on class loading and provides
 * methods for cleanup, specifically the
 * {@code close()} method to destroy the native resource pointer. Subclasses of
 * {@code CuFileHandle} should implement
 * specific functionality, such as reading or writing data, using this base
 * class to manage the native resource pointer.
 * </p>
 */
abstract class CuFileHandle implements AutoCloseable {
    private final long pointer;

    static {
        CuFile.initialize();
    }

    protected CuFileHandle(long pointer) {
        this.pointer = pointer;
    }

    public void close() {
        destroy(pointer);
    }

    protected long getPointer() {
        return this.pointer;
    }

    private static native void destroy(long pointer);
}
