import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

object MovieRecommenderTrainer {
  def main(args: Array[String]): Unit = {
    println("Initializing Spark...")
    val sparkConf = new SparkConf()
      .setAppName("MovieRecommenderTrainer")
      .set("spark.ui.showConsoleProgress", "false")
      .setMaster("local")

    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("WARN")
    println("Spark master: " + sc.master)

    println("Setting Up Logger...")
    setLogger()

    println("Setting Up Data Path...")
    val dataPath = "hdfs://localhost:9000/user/dharmesh/movierecommendation/ml-900MB/u.data"
    val modelPath = "hdfs://localhost:9000/user/dharmesh/movierecommendation/ml-900MB/ALSmodel"
    val checkpointPath = "hdfs://localhost:9000/user/dharmesh/ml-900MB/checkpoint/"
    sc.setCheckpointDir(checkpointPath)

    println("Preparing Data...")
    val ratingsRDD: RDD[Rating] = prepareData(sc, dataPath)
    ratingsRDD.checkpoint()

    println("Training ALS Model...")
    val model: MatrixFactorizationModel = trainALSModel(ratingsRDD)

    println("Saving Trained Model...")
    saveModel(sc, model, modelPath)

    sc.stop()
  }

  def setLogger(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
  }

  def prepareData(sc: SparkContext, dataPath: String): RDD[Rating] = {
    val file: RDD[String] = sc.textFile(dataPath)
    val ratingsRDD: RDD[Rating] = file.map(line => line.split("\t") match {
      case Array(user, item, rate, _) => Rating(user.toInt, item.toInt, rate.toDouble)
    })
    println("Sample Rating: " + ratingsRDD.first())
    ratingsRDD
  }

  def trainALSModel(ratingsRDD: RDD[Rating]): MatrixFactorizationModel = {
    println("Start ALS training, rank=5, iteration=20, lambda=0.1")
    ALS.train(ratingsRDD, 5, 20, 0.1)
  }

  def saveModel(context: SparkContext, model: MatrixFactorizationModel, modelPath: String): Unit = {
    try {
      model.save(context, modelPath)
    } catch {
      case _: Exception => println("Error occurred when saving the model.")
    }
  }
}
