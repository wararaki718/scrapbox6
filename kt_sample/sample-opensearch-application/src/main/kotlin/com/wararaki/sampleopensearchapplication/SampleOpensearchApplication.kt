package com.wararaki.sampleopensearchapplication

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.autoconfigure.data.elasticsearch.ElasticsearchDataAutoConfiguration
import org.springframework.boot.runApplication

@SpringBootApplication
class SampleOpensearchApplication

fun main(args: Array<String>) {
	runApplication<SampleOpensearchApplication>(*args)
}
