package bindings.kvikio.cufile;


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
