package com.wararaki.samplevectorsearch.repository

import com.wararaki.samplevectorsearch.model.Product
import org.springframework.data.elasticsearch.repository.ElasticsearchRepository
import org.springframework.stereotype.Repository

@Repository
interface SampleVectorSearchRepository : ElasticsearchRepository<Product, String> {
    fun findByNameLikeAndPriceGreaterThan(name: String, price: Int): List<Product>
}
