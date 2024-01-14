package com.wararaki.samplevectorsearch.model

import org.springframework.data.annotation.Id
import org.springframework.data.elasticsearch.annotations.Document
import org.springframework.data.elasticsearch.annotations.Field
import org.springframework.data.elasticsearch.annotations.FieldType
import org.springframework.data.elasticsearch.annotations.Mapping
import org.springframework.data.elasticsearch.annotations.Setting


@Document(indexName = "knn-index-sample")
@Setting(settingPath = "index-settings.json")
@Mapping(mappingPath = "index-mapping.json")
data class Product (
    @Id val id: String,
    @Field(type = FieldType.Text, name = "name") val name: String,
    @Field(type = FieldType.Double, name = "price") val price: Double,
    @Field(type = FieldType.Dense_Vector, name = "vector") val vector: List<Double>
) {}
