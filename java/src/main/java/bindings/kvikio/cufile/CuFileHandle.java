package bindings.kvikio.cufile;

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