/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.kvikio.cufile;

/**
 * The {@code CuFileHandle} class is an abstract base class for managing cuFile
 * resource handles in the native library.
 * It represents a common mechanism for interacting with files via cuFile,
 * storing the native pointer to a resource and
 * ensuring that the resource is properly destroyed when no longer needed.
 * <p>
 * This class initializes the cuFile library on class loading and provides
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
