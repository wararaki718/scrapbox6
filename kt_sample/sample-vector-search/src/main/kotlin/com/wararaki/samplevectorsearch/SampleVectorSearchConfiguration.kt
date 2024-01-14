package com.wararaki.samplevectorsearch

import com.wararaki.samplevectorsearch.model.Product
import com.wararaki.samplevectorsearch.repository.SampleVectorSearchRepository
import org.springframework.boot.ApplicationRunner
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class SampleVectorSearchConfiguration {
    @Bean
    fun databaseInitializer(repository: SampleVectorSearchRepository) = ApplicationRunner {
        val product1 = Product("sample1", "product1", 12.2, listOf(1.5, 2.5))
        repository.save(product1)

        val product2 = Product("sample2", "product2", 7.1, listOf(2.5, 3.5))
        repository.save(product2)

        val product3 = Product("sample3", "product3", 12.9, listOf(3.5, 4.5))
        repository.save(product3)

        val product4 = Product("sample4", "product4", 1.2, listOf(5.5, 6.5))
        repository.save(product4)

        val product5 = Product("sample5", "product5", 3.7, listOf(4.5, 5.5))
        repository.save(product5)
    }
}
