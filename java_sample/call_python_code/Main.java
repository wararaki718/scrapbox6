import org.graalvm.polyglot.*;

public class Main {
    public static void main(String[] args) {
        Context ctx = Context.newBuilder().allowAllAccess(true).build();
        try {
            File calculate = new File("./calculate.py");
            ctx.eval(Source.newBuilder("python", calculate).build());

            Value calculateFunction = ctx.getBindings("python").getMember("calculate");

            Value result = calculateFunction.execute(3, 5);
            System.out.println("Result: " + result.asInt());
        } catch (Exception e) {
            System.out.println("Error: " + e);
            e.printStackTrace();
        }
        System.out.println("DONE");
    }
}