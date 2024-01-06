package com.wararaki.sampleopensearchapplication.service

import com.wararaki.sampleopensearchapplication.model.Product
import com.wararaki.sampleopensearchapplication.repository.MarketplaceRepository
import org.springframework.stereotype.Service

@Service
class MarketplaceService() {
    private final lateinit var repository: MarketplaceRepository

    fun findByName(name: String): List<Product> {
        return repository.findByNameLikeAndPriceGreaterThan(name, 0)
    }

    fun insertProduct(name: String): Product {
        var product = Product("id_{name}", name, 100)
        repository.save(product)
        return product
    }
}
