package com.lucaswonderley.consume

import org.json4s._
import org.json4s.jackson.Serialization.{read,write}
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.flume._
import scala.collection.immutable.StringOps
import org.apache.log4j.{Level, LogManager, PropertyConfigurator}

case class NewsArticle(article_id : String, url : String, title : String, feed_id : String, vec : String)

object ConsumeNewsStream {
    def main(args : Array[String]) = {
        val conf = new SparkConf().setAppName("ConsumeNewsStream")
        val ssc = new StreamingContext(conf, Seconds(30))
        val log = LogManager.getRootLogger 
        log.warn("Test log in ConsumeNewsStream")
        val flumeStream = FlumeUtils.createStream(ssc, "localhost", 44444)
        val articleData = flumeStream.map(record => {
            implicit val formats = DefaultFormats
            read[NewsArticle](new String(record.event.getBody().array()))
        })
        articleData.print()
        articleData.count().map(cnt => "Received" + cnt + " events").print()
        val curTime: Long = System.currentTimeMillis
        articleData.foreachRDD(r => {
            if (r.count() > 0) {
                r.map(article => {
                    implicit val formats = DefaultFormats
                    write(article)
                }).saveAsTextFile("/home/ec2-user/news_agg/news_stream-" + curTime.toString())
            }
        })
        ssc.start()
        ssc.awaitTermination()
    }
}
    
