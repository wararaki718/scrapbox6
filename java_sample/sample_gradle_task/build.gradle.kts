tasks.register("hello") {
    doLast {
        println("hello!")
    }
}

tasks.register("greet") {
    println("greet")
    dependsOn("hello")
    doLast {
        println("world")
    }
}
