plugins {
    `cpp-application`
}

group = "com.wararaki"
version = "1.0-SNAPSHOT"

//application {
//    dependencies {
//        implementation(project(":common"))
//    }
//}

application {
    source.from(file("src"))
    privateHeaders.from(file("src"))
}



tasks.withType(CppCompile::class.java).configureEach {
//    macros.put("NDEBUG", null)

    compilerArgs.add("-W3")

    compilerArgs.addAll(toolChain.map {
        toolChain -> when (toolChain) {
            is Gcc, is Clang -> listOf("-O2", "-fno-access-control")
            is VisualCpp -> listOf("/Zi")
            else -> listOf()
        }
    })
}