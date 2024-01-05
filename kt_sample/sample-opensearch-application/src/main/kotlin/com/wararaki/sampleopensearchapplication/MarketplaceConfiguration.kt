package com.wararaki.sampleopensearchapplication

import com.wararaki.sampleopensearchapplication.repository.MarketplaceRepository
import org.apache.http.conn.ssl.TrustSelfSignedStrategy
import org.apache.http.impl.nio.client.HttpAsyncClientBuilder
import org.apache.http.ssl.SSLContextBuilder
import org.opensearch.spring.boot.autoconfigure.RestClientBuilderCustomizer
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.ComponentScan
import org.springframework.context.annotation.Configuration
import org.springframework.data.elasticsearch.repository.config.EnableElasticsearchRepositories

@Configuration
@EnableElasticsearchRepositories(basePackageClasses = [MarketplaceRepository::class])
@ComponentScan(basePackageClasses = [MarketplaceConfiguration::class])
class MarketplaceConfiguration {
    @Bean
    fun customizer(): RestClientBuilderCustomizer {
        return RestClientBuilderCustomizer(){
            fun customize(builder: HttpAsyncClientBuilder) {
                try {
                    builder.setSSLContext(
                        SSLContextBuilder().loadTrustMaterial(null, TrustSelfSignedStrategy()).build()
                    )
                } catch (e: Exception) {
                    throw RuntimeException("Failed to initialize SSL Context instance", e)
                }
            }
        }
    }
}