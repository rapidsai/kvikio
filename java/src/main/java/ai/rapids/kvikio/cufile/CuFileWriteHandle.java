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
 * The {@code CuFileWriteHandle} class represents a handle for writing data to a
 * file using the cuFile library.
 * <p>
 * This class is initialized with the path to the file to be written to, and it
 * creates a native resource pointer
 * for interacting with the file. The {@code write()} method allows writing data
 * from a device buffer to the file at
 * a specified location and offset.
 * </p>
 */
public final class CuFileWriteHandle extends CuFileHandle {

    /**
     * Constructs a {@code CuFileWriteHandle} for the specified file path.
     *
     * @param path The path to the file to be opened for writing.
     */
    public CuFileWriteHandle(String path) {
        super(create(path));
    }

    /**
     * Writes data from a device buffer to the file represented by this handle.
     *
     * @param device_pointer The pointer to the device buffer containing the data to
     *                       write.
     * @param size           The number of bytes to write.
     * @param file_offset    The offset in the file where to start writing.
     * @param buffer_offset  The offset in the device buffer to start reading data.
     */
    public void write(long device_pointer, long size, long file_offset, long buffer_offset) {
        writeFile(getPointer(), device_pointer, size, file_offset, buffer_offset);
    }

    private static native long create(String path);

    private static native void writeFile(long file_pointer, long device_pointer, long size, long file_offset,
            long buffer_offset);
}
