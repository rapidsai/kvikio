/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
