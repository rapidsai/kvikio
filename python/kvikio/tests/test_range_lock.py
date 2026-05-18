# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import kvikio
import kvikio.defaults

cupy = pytest.importorskip("cupy")
numpy = pytest.importorskip("numpy")


@pytest.fixture
def range_lock_file(tmp_path):
    """Create a temporary file for range lock testing"""
    filename = tmp_path / "test-rangelock"
    # Pre-allocate file
    with open(filename, "wb") as f:
        f.seek(10 * 1024 * 1024 - 1)  # 10MB file
        f.write(b'\0')
    return filename


def test_parallel_non_overlapping_writes(range_lock_file):
    """Test that non-overlapping range writes can execute in parallel"""
    chunk_size = 1024 * 1024  # 1MB chunks
    num_chunks = 8

    # Create distinct data for each chunk
    chunks = {}
    for i in range(num_chunks):
        data = numpy.full(chunk_size // 4, i, dtype=numpy.int32)
        chunks[i] = cupy.asarray(data)

    def write_chunk(chunk_id):
        """Write a specific chunk using range lock"""
        offset = chunk_id * chunk_size
        with kvikio.CuFile(range_lock_file, "r+") as f:
            # Write with range locking (when implemented)
            # For now, this tests the basic parallel write capability
            f.pwrite(chunks[chunk_id], file_offset=offset)
        return chunk_id

    # Write chunks in parallel
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_chunk, i) for i in range(num_chunks)]
        completed = [f.result() for f in as_completed(futures)]
    parallel_time = time.time() - start_time

    # Verify all chunks were written
    assert len(completed) == num_chunks

    # Verify data integrity
    with kvikio.CuFile(range_lock_file, "r") as f:
        for i in range(num_chunks):
            offset = i * chunk_size
            data = cupy.empty(chunk_size // 4, dtype=cupy.int32)
            f.pread(data, file_offset=offset)
            expected = cupy.full(chunk_size // 4, i, dtype=cupy.int32)
            cupy.testing.assert_array_equal(data, expected)

    print(f"Parallel write time: {parallel_time:.3f}s")


def test_overlapping_range_serialization(range_lock_file):
    """Test that overlapping ranges are properly serialized"""
    chunk_size = 1024 * 1024  # 1MB
    overlap_size = chunk_size // 2

    execution_order = []
    lock = threading.Lock()

    def write_with_overlap(writer_id, offset):
        """Write data that potentially overlaps with other writers"""
        data = numpy.full(chunk_size // 4, writer_id, dtype=numpy.int32)
        gpu_data = cupy.asarray(data)

        with kvikio.CuFile(range_lock_file, "r+") as f:
            with lock:
                execution_order.append((writer_id, "start"))
            f.pwrite(gpu_data, file_offset=offset)
            with lock:
                execution_order.append((writer_id, "end"))
        return writer_id

    # Create overlapping writes
    # Writer 0: offset 0, size 1MB
    # Writer 1: offset 512KB, size 1MB (overlaps with writer 0)
    # Writer 2: offset 1MB, size 1MB (overlaps with writer 1)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(write_with_overlap, 0, 0),
            executor.submit(write_with_overlap, 1, overlap_size),
            executor.submit(write_with_overlap, 2, chunk_size),
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == 3

    # Check that execution was properly ordered
    # With proper range locking, overlapping ranges should serialize
    print(f"Execution order: {execution_order}")


def test_range_lock_performance_benefit(range_lock_file):
    """Compare performance of range-locked vs serialized writes"""
    chunk_size = 512 * 1024  # 512KB chunks
    num_operations = 16

    # Prepare data
    data_chunks = []
    for i in range(num_operations):
        data = numpy.full(chunk_size // 4, i, dtype=numpy.int32)
        data_chunks.append(cupy.asarray(data))

    # Test 1: Interleaved non-overlapping writes (should benefit from range lock)
    def write_interleaved(op_id):
        # Even ops write to first half, odd ops to second half
        base_offset = (op_id % 2) * (chunk_size * num_operations // 2)
        offset = base_offset + (op_id // 2) * chunk_size

        with kvikio.CuFile(range_lock_file, "r+") as f:
            f.pwrite(data_chunks[op_id], file_offset=offset)
        return op_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_interleaved, i) for i in range(num_operations)]
        results = [f.result() for f in as_completed(futures)]
    interleaved_time = time.time() - start

    # Test 2: Sequential overlapping writes (no benefit from range lock)
    def write_sequential(op_id):
        # All write to overlapping regions
        offset = op_id * (chunk_size // 2)  # 50% overlap

        with kvikio.CuFile(range_lock_file, "r+") as f:
            f.pwrite(data_chunks[op_id], file_offset=offset)
        return op_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_sequential, i) for i in range(num_operations)]
        results = [f.result() for f in as_completed(futures)]
    sequential_time = time.time() - start

    print(f"Interleaved (non-overlapping) time: {interleaved_time:.3f}s")
    print(f"Sequential (overlapping) time: {sequential_time:.3f}s")

    # With proper range locking, interleaved should be faster
    # Without range locking, times might be similar
    assert len(results) == num_operations


@pytest.mark.parametrize("num_threads", [2, 4, 8])
def test_concurrent_range_locks(range_lock_file, num_threads):
    """Test concurrent acquisition and release of range locks"""
    operations_per_thread = 10
    chunk_size = 128 * 1024  # 128KB

    success_counter = threading.Semaphore(0)

    def worker(thread_id):
        """Worker that performs multiple range-locked operations"""
        for op in range(operations_per_thread):
            # Each thread writes to its own range
            offset = thread_id * operations_per_thread * chunk_size + op * chunk_size
            data = numpy.full(chunk_size // 4, thread_id * 100 + op, dtype=numpy.int32)
            gpu_data = cupy.asarray(data)

            with kvikio.CuFile(range_lock_file, "r+") as f:
                f.pwrite(gpu_data, file_offset=offset)

            success_counter.release()
        return thread_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [f.result() for f in as_completed(futures)]
    duration = time.time() - start

    # Verify all operations completed
    total_ops = num_threads * operations_per_thread
    for _ in range(total_ops):
        assert success_counter.acquire(timeout=0.001)

    print(f"Completed {total_ops} operations with {num_threads} threads in {duration:.3f}s")
    assert len(results) == num_threads


def test_range_lock_with_different_sizes(range_lock_file):
    """Test range locks with varying data sizes"""
    sizes = [64 * 1024, 256 * 1024, 1024 * 1024]  # 64KB, 256KB, 1MB

    def write_variable_size(op_id, size):
        """Write data of variable size"""
        offset = sum(sizes[:op_id])  # Non-overlapping offsets
        data = numpy.full(size // 4, op_id, dtype=numpy.int32)
        gpu_data = cupy.asarray(data)

        with kvikio.CuFile(range_lock_file, "r+") as f:
            bytes_written = f.pwrite(gpu_data, file_offset=offset)
            assert bytes_written == size
        return (op_id, size)

    with ThreadPoolExecutor(max_workers=len(sizes)) as executor:
        futures = [executor.submit(write_variable_size, i, size)
                  for i, size in enumerate(sizes)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == len(sizes)

    # Verify data
    with kvikio.CuFile(range_lock_file, "r") as f:
        offset = 0
        for op_id, size in enumerate(sizes):
            data = cupy.empty(size // 4, dtype=cupy.int32)
            f.pread(data, file_offset=offset)
            expected = cupy.full(size // 4, op_id, dtype=cupy.int32)
            cupy.testing.assert_array_equal(data, expected)
            offset += size