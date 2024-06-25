package bindings.kvikio.cufile;

public final class CuFileReadHandle extends CuFileHandle{
    
    public CuFileReadHandle(String path) {
        super(create(path));
    }

    public void read(long device_pointer, long size, long file_offset, long device_offset) {
        readFile(getPointer(),device_pointer,size,file_offset,device_offset);
    }

    private static native long create(String path);

    private static native void readFile(long file_pointer, long device_pointer, long size, long file_offset, long device_offset);
    
}
