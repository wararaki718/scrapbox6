package call_python;

import java.io.File;
import org.graalvm.polyglot.*;


public class App {
    public static void main(String[] args) {
        Context context = Context.newBuilder().allowAllAccess(true).build();
        try {
            File program = new File(ClassLoader.getSystemClassLoader().getResource("./calculate.py").toURI());
            context.eval(Source.newBuilder("python", program).build());

            Value calculateFunction = context.getBindings("python").getMember("calculate");
            Value result = calculateFunction.execute(3, 5);
            System.out.println(result.asInt());
        } catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }
        System.out.println("DONE");
    }
}
