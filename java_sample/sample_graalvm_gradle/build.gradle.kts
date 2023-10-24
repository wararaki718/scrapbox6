plugins {
    id("java")
    id("org.graalvm.buildtools.native") version "0.9.27"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}

graalvmNative {
    toolchainDetection.set(true)
//    binaries {
//        named("main") {
//            javaLauncher.set(javaToolchains.launcherFor {
//                languageVersion.set(JavaLanguageVersion.of(17))
//                vendor.set(JvmVendorSpec.matching("Oracle Corporation"))
//            })
//        }
//    }
}
