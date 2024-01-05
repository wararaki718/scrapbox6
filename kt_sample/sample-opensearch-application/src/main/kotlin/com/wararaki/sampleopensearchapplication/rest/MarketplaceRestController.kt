package com.wararaki.sampleopensearchapplication.rest

import com.wararaki.sampleopensearchapplication.model.Product
import com.wararaki.sampleopensearchapplication.repository.MarketplaceRepository
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.ResponseBody
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/marketplace")
class MarketplaceRestController {
    lateinit var repository: MarketplaceRepository

    @GetMapping(value = ["/search"], produces = [MediaType.APPLICATION_JSON_VALUE])
    @ResponseBody
    fun search(
        @RequestParam(value = "name", required = false, defaultValue = "") name: String,
        @RequestParam(value = "price", required = false, defaultValue = "0.0") price: Double
    ): List<Product> {
        return repository.findByNameLikeAndPriceGreaterThan(name, price)
    }
}