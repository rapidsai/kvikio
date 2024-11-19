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
 * The {@code CuFileReadHandle} class represents a handle for reading data from
 * a file using the cuFile library.
 * <p>
 * This class is initialized with the path to the file to be read, and it
 * creates a native resource pointer for
 * interacting with the file. The {@code read()} method allows reading data from
 * the file at a specified location and
 * offset, transferring it to a device buffer.
 * </p>
 */
public final class CuFileReadHandle extends CuFileHandle {

    /**
     * Constructs a {@code CuFileReadHandle} for the specified file path.
     *
     * @param path The path to the file to be opened for reading.
     */
    public CuFileReadHandle(String path) {
        super(create(path));
    }

    /**
     * Reads data from the file represented by this handle into a device buffer.
     *
     * @param device_pointer The pointer to the device buffer where the data will be
     *                       written.
     * @param size           The number of bytes to read.
     * @param file_offset    The offset in the file from where to start reading.
     * @param device_offset  The offset in the device buffer to start writing data.
     */
    public void read(long device_pointer, long size, long file_offset, long device_offset) {
        readFile(getPointer(), device_pointer, size, file_offset, device_offset);
    }

    private static native long create(String path);

    private static native void readFile(long file_pointer, long device_pointer, long size, long file_offset,
            long device_offset);

}
