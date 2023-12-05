import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.log4j.{Logger, Level}
import org.apache.spark.rdd.RDD

object MovieRecommender {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Please provide 2 parameters:")
      println("1. Recommendation type: '--U' for user-based recommendation; '--M' for movie-based recommendation.")
      println("2. Input ID: UserID ('--U'); MovieID ('--M')")
      sys.exit(0)
    }

    val recommendationType = args(0)
    val inputID = args(1)

    println("Initializing Spark...")
    val sparkConf = new SparkConf()
      .setAppName("MovieRecommender")
      .set("spark.ui.showConsoleProgress", "false")
      .setMaster("local")
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("WARN")
    println("Spark master: " + sc.master)

    println("Setting Up Logger...")
    setLogger()

    println("Setting Up Data Path...")
    val dataPath = "hdfs://localhost:9000/user/dharmesh/movierecommendation/ml-900MB/u.item"
    val modelPath = "hdfs://localhost:9000/user/dharmesh/movierecommendation/ml-900MB/ALSmodel"
    val checkpointPath = "hdfs://localhost:9000/user/dharmesh/ml-900MB/checkpoint/"
    sc.setCheckpointDir(checkpointPath)

    println("Preparing Data...")
    val movieTitle: RDD[(Int, String)] = loadData(sc, dataPath)
    movieTitle.checkpoint()

    println("Loading Model...")
    val model = loadModel(sc, modelPath)

    println("Making Recommendations...")
    recommend(model, movieTitle, recommendationType, inputID)

    sc.stop()
  }

  def setLogger(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
  }

  def loadData(sc: SparkContext, dataPath: String): RDD[(Int, String)] = {
    println("Loading Data...")
    val itemRDD: RDD[String] = sc.textFile(dataPath)
    val movieTitle: RDD[(Int, String)] = itemRDD.map(line => line.split("\\|")).map(x => (x(0).toInt, x(1)))
    movieTitle
  }

  def loadModel(sc: SparkContext, modelPath: String): Option[MatrixFactorizationModel] = {
    try {
      val model: MatrixFactorizationModel = MatrixFactorizationModel.load(sc, modelPath)
      Some(model)
    } catch {
      case _: Exception => None
    }
  }

  def recommend(model: Option[MatrixFactorizationModel], movieTitle: RDD[(Int, String)], arg1: String, arg2: String): Unit = {
    if (arg1 == "--U") {
      recommendMovies(model.get, movieTitle, arg2.toInt)
    }
    if (arg1 == "--M") {
      recommendUsers(model.get, movieTitle, arg2.toInt)
    }
  }

  def recommendMovies(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputUserID: Int): Unit = {
    val recommendations = model.recommendProducts(inputUserID, 20)
    println(s"Top 20 movie recommendations for user $inputUserID:")
    recommendations.foreach { r =>
      val movieName = movieTitle.lookup(r.product).mkString
      println(s"Movie: $movieName, Rating: ${r.rating}")
    }
  }

  def recommendUsers(model: MatrixFactorizationModel, movieTitle: RDD[(Int, String)], inputMovieID: Int): Unit = {
    val recommendations = model.recommendUsers(inputMovieID, 20)
    println(s"Top 20 user recommendations for movie $inputMovieID:")
    recommendations.foreach { r =>
      val movieName = movieTitle.lookup(r.product).mkString
      println(s"Movie: $movieName, Recommended User: ${r.user}, Rating: ${r.rating}")
    }
  }
}
