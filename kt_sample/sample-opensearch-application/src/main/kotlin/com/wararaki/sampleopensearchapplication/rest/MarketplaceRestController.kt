package com.wararaki.sampleopensearchapplication.rest

import com.wararaki.sampleopensearchapplication.model.Product
import com.wararaki.sampleopensearchapplication.repository.MarketplaceRepository
import com.wararaki.sampleopensearchapplication.service.MarketplaceService
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping("/marketplace")
class MarketplaceRestController {
    var marketplaceService = MarketplaceService()

    @GetMapping(value = ["/search"], produces = [MediaType.APPLICATION_JSON_VALUE])
    @ResponseBody
    fun search(
        @RequestParam(value = "name", required = false, defaultValue = "") name: String,
    ): List<Product> {
        return marketplaceService.findByName(name)
    }

    @PostMapping(value = ["/insert"])
    fun insert(
        @RequestParam(value = "name", required = true) name: String,
    ): Product {
        return marketplaceService.insertProduct(name)
    }
}
