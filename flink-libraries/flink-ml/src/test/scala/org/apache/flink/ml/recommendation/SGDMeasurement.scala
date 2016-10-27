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
import org.apache.flink.ml.util.FlinkTestBase
import org.scalatest._

import scala.language.postfixOps

object SGDMeasurement {

  def main(args: Array[String]): Unit = {
    val parallelism = 2

    import Recommendation._

    val env = ExecutionEnvironment.getExecutionEnvironment

    val dsgd = SGD()
      .setIterations(20)
      .setLambda(0)
      .setBlocks(3)
      .setNumFactors(10)
      .setLearningRate(0.001)
      .setSeed(43L)

    val pathToTrainingFile = "/home/dani/data/movielens_train.csv"
    // Read input data set from a csv file
    val inputDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](pathToTrainingFile)

    dsgd.fit(inputDS)


    val pathToData = "/home/dani/data/movielens_test.csv"
    val testingDS = env.readCsvFile[(Int, Int)](pathToData).first(10)

    val predictions = dsgd.predict(testingDS).collect()

    //  val userFacts = dsgd.factorsOption.get._1.collect
    //  val itemFacts = dsgd.factorsOption.get._2.collect
    predictions.foreach(println)
    println("------------------")

  }


}
