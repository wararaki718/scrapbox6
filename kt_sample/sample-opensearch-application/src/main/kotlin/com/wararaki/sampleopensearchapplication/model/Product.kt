package com.wararaki.sampleopensearchapplication.model

import org.springframework.data.annotation.Id
import org.springframework.data.elasticsearch.annotations.Document
import org.springframework.data.elasticsearch.annotations.Field
import org.springframework.data.elasticsearch.annotations.FieldType

@Document(indexName = "marketplace")
data class Product(
    @Id val id: String,
    @Field(type = FieldType.Text, name = "name") val name: String,
    @Field(type = FieldType.Double, name = "price") val price: Double,
    @Field(type = FieldType.Integer, name = "quantity") val quantity: Int,
    @Field(type = FieldType.Text, name = "description") val description: String,
    @Field(type = FieldType.Keyword, name = "vendor") val vendor: String,
){}
