package com.wararaki.sampleopensearchapplication.repository

import com.wararaki.sampleopensearchapplication.model.Product
import org.springframework.data.elasticsearch.repository.ElasticsearchRepository
import org.springframework.stereotype.Repository

@Repository
interface MarketplaceRepository : ElasticsearchRepository<Product, String> {
    fun findByNameLikeAndPriceGreaterThan(name: String, price: Int): List<Product>
}
