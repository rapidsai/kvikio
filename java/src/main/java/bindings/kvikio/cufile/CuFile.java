package bindings.kvikio.cufile;

public class CuFile {
    private static boolean initialized = false;
    private static CuFileDriver driver;

    static {
        initialize();
    }

    static synchronized void initialize() {
        if (!initialized) {
            try {
                System.loadLibrary("CuFileJNI");
                driver = new CuFileDriver();
                Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                    driver.close();
                  }));
                initialized = true;
            } catch (Throwable t) {
                System.out.println("could not load cufile jni library");
            }
        }
    }

    public static boolean libraryLoaded() {
        return initialized;
    }
}
