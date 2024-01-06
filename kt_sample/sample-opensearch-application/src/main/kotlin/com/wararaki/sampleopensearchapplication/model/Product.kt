package com.wararaki.sampleopensearchapplication.model

import org.springframework.data.annotation.Id
import org.springframework.data.elasticsearch.annotations.Document
import org.springframework.data.elasticsearch.annotations.Field
import org.springframework.data.elasticsearch.annotations.FieldType

@Document(indexName = "marketplace")
data class Product(
    @Id val id: String,
    @Field(type = FieldType.Text, name = "name") val name: String,
    @Field(type = FieldType.Integer, name = "price") val price: Int,
){}
