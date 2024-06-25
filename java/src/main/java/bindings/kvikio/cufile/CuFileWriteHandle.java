package bindings.kvikio.cufile;

public final class CuFileWriteHandle extends CuFileHandle {
    
    public CuFileWriteHandle(String path) {
        super(create(path));
    }

    public void write(long device_pointer, long size, long file_offset, long buffer_offset) {
        writeFile(getPointer(),device_pointer,size,file_offset,buffer_offset);
    }

    private static native long create(String path);

    private static native void writeFile(long file_pointer, long device_pointer, long size, long file_offset, long buffer_offset);
}
