/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package call_numpy;

import java.nio.file.Paths;
import java.io.File;
import org.graalvm.polyglot.*;


public class App {
    public static void main(String[] args) {
        // String venvExePath = App.class.getClassLoader().getResource(Paths.get("venv", "bin", "graalpy").toString()).getPath();
        String venvExePath = Paths.get("venv", "bin", "graalpy").toString();
        Context context = Context.newBuilder()
            .allowAllAccess(true)
            .option("python.ForceImportSite", "true")
            .option("python.Executable", venvExePath)
            .build();

        try {
            File program = new File(ClassLoader.getSystemClassLoader().getResource("./calculate.py").toURI());
            context.eval(Source.newBuilder("python", program).build());

            int[] a = {1, 2};
            int[] b = {2, 3};

            Value calculateFunction = context.getBindings("python").getMember("calculate");
            Value result = calculateFunction.execute(a, b);
            System.out.println(result.getArrayElement(0));
        } catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        System.out.println("DONE");
    }
}
