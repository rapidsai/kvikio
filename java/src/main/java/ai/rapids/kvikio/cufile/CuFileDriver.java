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
 * The {@code CuFileDriver} class provides an interface for interacting with the
 * cuFile native library, specifically
 * for managing the lifecycle of the CuFileDriver.
 * <p>
 * This class is responsible for allocating and deallocating the native
 * resources needed by the cuFile API. The
 * {@code pointer} field represents the native resource that is created when the
 * driver is instantiated and
 * destroyed when the driver is closed. The class ensures proper cleanup of
 * native resources to avoid memory leaks.
 * </p>
 */
final class CuFileDriver implements AutoCloseable {
    private final long pointer;

    CuFileDriver() {
        pointer = create();
    }

    public void close() {
        destroy(pointer);
    }

    private static native long create();

    private static native void destroy(long pointer);
}
