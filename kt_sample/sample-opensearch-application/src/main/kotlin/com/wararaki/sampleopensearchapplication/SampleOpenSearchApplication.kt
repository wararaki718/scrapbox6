package com.wararaki.sampleopensearchapplication

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.autoconfigure.data.elasticsearch.ElasticsearchDataAutoConfiguration
import org.springframework.boot.runApplication
import org.springframework.data.elasticsearch.repository.config.EnableElasticsearchRepositories

@EnableElasticsearchRepositories
@SpringBootApplication(exclude = [ElasticsearchDataAutoConfiguration::class])
class SampleOpenSearchApplication

fun main(args: Array<String>) {
	runApplication<SampleOpenSearchApplication>(*args)
}
