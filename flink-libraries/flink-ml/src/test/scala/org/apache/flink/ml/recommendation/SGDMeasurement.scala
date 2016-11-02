/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.recommendation

import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem
import org.apache.flink.ml.util.FlinkTestBase
import org.scalatest._

import scala.language.postfixOps

object SGDMeasurement {

  def main(args: Array[String]): Unit = {

    def getRmse(iterations: Int, lambda: Int, numBlocks: Int, numFactors: Int, learningRate: Double,
                seed: Long, trainDS: DataSet[(Int, Int, Double)], testDS: DataSet[(Int, Int, Double)]) = {

      val dsgd = SGD()
        .setIterations(iterations)
        .setLambda(lambda)
        .setBlocks(numBlocks)
        .setNumFactors(numFactors)
        .setLearningRate(learningRate)
        .setSeed(seed)

      dsgd.fit(trainDS)

      val testWithoutRatings = testDS.map(i => (i._1, i._2))

      val predictions = dsgd.predict(testWithoutRatings).collect()

      val environment = trainDS.getExecutionEnvironment
      val predDS = environment.fromCollection(predictions)

      val rmse = testDS.join(predDS).where(0).equalTo(0)
//        .map(i => (i._1._3, i._2._3))
//        .map(i => (i._1 - i._2) * (i._1 - i._2))
//        .map(i => (i, 1))
//        .reduce((i, j) => (i._1 + i._1, i._2 + j._2))
//        .map(i => math.sqrt(i._1 / i._2))

/*      println("*****************************************************************")
      rmse.print()
      //testDS.print()
      println("*****************************************************************")
      //rmse.collect().iterator.next()
      9999.999*/

      rmse
    }

    val trainPath = "/home/dani/data/movielens_train.csv"
    val testPath = "/home/dani/data/movielens_test.csv"
    val predPath = "/home/dani/data/movielens_pred.csv"

    val env = ExecutionEnvironment.getExecutionEnvironment

    val trainDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](trainPath)
    val testDS = env.readCsvFile[(Int, Int, Double)](testPath).first(15)
    val predDS = env.readCsvFile[(Int, Int, Double)](predPath).first(15)

    //val testRmse = getRmse(20, 0, 4, 10, 0.01, 43L, trainDS, testDS)

    //println(testRmse)
    val joined = testDS.join(predDS).where(0, 1).equalTo(0, 1)
      .map(i => (i._1._3, i._2._3))
//      .map(i => ((i._1 - i._2) * (i._1 - i._2), 1))
//      .map(i => (i, 1))
//      .reduce((i, j) => (i._1 + i._1, i._2 + j._2))
//      .map(i => math.sqrt(i._1 / i._2))

    joined.writeAsCsv("/home/dani/data/tmp/joined.csv",
      writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
//
//    joined foreach(println(_))
    joined.print()


/*    val proba = env.fromCollection(List((2, 1, -1), (1, 5, -2), (8, 7, -3), (5, 1, -4), (2, 3, -5)))
    val proba2 = env.fromCollection(List((2, 1, 1), (1, 5, 2), (8, 7, 3), (5, 1, 4), (2, 3, 5)))

    proba.join(proba2).where(0, 1).equalTo(0, 1).print()



    proba.join(proba2).where(0, 1).equalTo(0, 1).writeAsCsv("/home/dani/data/tmp/joined.csv",
      writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)*/
  }


}
