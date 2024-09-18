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

public final class CuFileWriteHandle extends CuFileHandle {

    public CuFileWriteHandle(String path) {
        super(create(path));
    }

    public void write(long device_pointer, long size, long file_offset, long buffer_offset) {
        writeFile(getPointer(), device_pointer, size, file_offset, buffer_offset);
    }

    private static native long create(String path);

    private static native void writeFile(long file_pointer, long device_pointer, long size, long file_offset,
            long buffer_offset);
}
