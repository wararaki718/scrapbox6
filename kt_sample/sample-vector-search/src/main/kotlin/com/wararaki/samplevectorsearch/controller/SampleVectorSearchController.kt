package com.wararaki.samplevectorsearch.controller

import com.wararaki.samplevectorsearch.model.Product
import com.wararaki.samplevectorsearch.model.VectorQuery
import com.wararaki.samplevectorsearch.service.SampleVectorSearchService
import org.springframework.data.elasticsearch.core.SearchHits
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.*

@RestController
@RequestMapping("/vector")
class SampleVectorSearchController (private val service: SampleVectorSearchService){
    @PostMapping(value = ["/search"], produces = [MediaType.APPLICATION_JSON_VALUE])
    @ResponseBody
    fun search(@RequestBody query: VectorQuery): SearchHits<Product> {
        return service.vectorSearch(query.vector)
    }
}
