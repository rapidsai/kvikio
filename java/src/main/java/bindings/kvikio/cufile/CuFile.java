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
