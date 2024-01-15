package com.wararaki.samplevectorsearch.service

import com.wararaki.samplevectorsearch.model.Product
import com.wararaki.samplevectorsearch.repository.SampleVectorSearchRepository
import org.springframework.data.elasticsearch.core.ElasticsearchOperations
import org.springframework.data.elasticsearch.core.SearchHits
import org.springframework.data.elasticsearch.core.query.StringQuery
import org.springframework.stereotype.Service

@Service
class SampleVectorSearchService(private val operations: ElasticsearchOperations) {
    fun vectorSearch(vector: List<Double>): SearchHits<Product> {
        val strVector = vector.joinToString(prefix = "[", postfix = "]")
        val query = StringQuery("{ \"knn\":{\"vector\": {\"vector\": $strVector, \"k\": 2}}}")
        return operations.search(query, Product::class.java)
    }
}
