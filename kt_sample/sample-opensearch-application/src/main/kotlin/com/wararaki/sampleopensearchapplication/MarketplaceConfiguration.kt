package com.wararaki.sampleopensearchapplication

import com.wararaki.sampleopensearchapplication.model.Product
import com.wararaki.sampleopensearchapplication.repository.MarketplaceRepository
import org.springframework.boot.ApplicationRunner
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

@Configuration
class MarketplaceConfiguration {
    @Bean
    fun databaseInitializer(marketplaceRepository: MarketplaceRepository) = ApplicationRunner {
        val product = Product("sample1", "sample_name", 10000)
        marketplaceRepository.save(product)
        val product2 = Product("sample2", "test_name", 1200)
        marketplaceRepository.save(product2)
    }
}