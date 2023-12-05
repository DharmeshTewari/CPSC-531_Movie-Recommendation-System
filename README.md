![](imgs/banner.jpg)

# A movie recommendation system built using Ubantu, Scala, Spark, and Hadoop

## Table of Contents

- [A movie recommendation system built using Scala, Spark and Hadoop](#a-movie-recommendation-system-built-using-scala-spark-and-hadoop)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction to movie recommendation system](#1-introduction-to-movie-recommendation-system)
    - [1.1 Different recommendataion system algorithms](#11-different-recommendataion-system-algorithms)
    - [1.2 Collaborative filtering and Spark ALS](#12-collaborative-filtering-and-spark-als)
  - [2. System setup](#2-system-setup)
  - [3. Dataset](#3-dataset)
  - [4. Running in Spark](#4-running-in-spark)
    - [4.1 Clone code from Github](#41-clone-code-from-github)
    - [4.2 Preparing data in HDFS](#42-preparing-data-in-hdfs)
    - [4.3 Train Movie recommendataion model in Spark](#43-train-movie-recommendataion-model-in-spark)
    - [4.4 Generating recommendations in Spark](#44-generating-recommendations-in-spark)
  - [5. Movie Recommendation system design](#5-movie-recommendation-system-design)
  - [6. Implementation](#6-implementation)
    - [6.1 Training ALS model - `MovieRecommenderTrainer.scala`](#61-training-als-model---movierecommendertrainerscala)
      - [6.1.1 `prepareData`](#611-preparedata)
      - [6.1.2 `ALS.train`](#612-alstrain)
      - [6.1.3 `saveModel`](#613-savemodel)
    - [6.2 Generating recommendations - `MovieRecommender.scala`](#62-generating-recommendations---movierecommenderscala)
      - [6.2.1 `prepareData`](#621-preparedata)
      - [6.2.2 `MatrixFactorizationModel.load`](#622-matrixfactorizationmodelload)
      - [6.2.3 `model.recommendProducts`](#623-modelrecommendproducts)
      - [6.2.4 `model.recommendUsers`](#624-modelrecommendusers)
  - [7. Summary](#7-summary)
  - [Contact](#contact)

A recommendation system is a highly utilized machine learning technique with diverse applications in E-commerce platforms such as Amazon and Alibaba, video streaming services like Netflix and Disney+, and social networks including Facebook and LinkedIn. Given the substantial volume of data in these services, contemporary industry-level recommendation systems are commonly constructed within big data frameworks like Spark and Hadoop. In this blog, I aim to share my experience in developing a movie recommendation system using Scala, Spark, and Hadoop.

## 1. Introduction to movie recommendation system
### 1.1 Different recommendataion system algorithms
Recommendation system algorithms can be broadly classified into two main types: content-based recommendation and collaborative filtering. The following summary table outlines the distinctions between these two approaches.

| |Content-based recommendataion | Collaborative filtering |
|--|-----------|--------------|
|Description|Utilizes product characteristics to recommend similar products to what a user previously liked.|Predicts the interest of a user by collecting preference information from many other users.|
|Assumption|If person P1 and person P2 have the same opinion on product D1, then P1 is more likely to have the same opinion on product D2 with P2 than with a random chosen person Px.|If person P likes product D1 which has a collection of attributes, he/she is more likely to like product D2 which shares those attributes than product D3 which doesn't.|
|Example|news/article recommendataion|movie recommendataion, Amazon product recommendation|
|Advantages| - The model doesn't need any user data input, so easier to scale.<br /> - Capable of catching niche items with feature engineering.| - No domain knowledge needed, highly transferrable model.<br /> - Capable of helping users discover new interests.|
|Disadvantages| - Requires domain knowledge.<br /> - Limited ability to expand user's interests.| - Cold-start problem: need to work with existing data, can't handle fresh items/users.<br /> - Difficulty in expanding features for items.|

### 1.2 Collaborative filtering and Spark ALS
In this article, we will employ collaborative filtering as the recommendation algorithm. The mechanism of collaborative filtering involves initially treating the ratings given by all users to all items as a matrix. This matrix can be decomposed into two distinct matrices: one representing users, where rows denote users and columns signify latent factors, and the other representing items, with rows corresponding to latent factors and columns representing items (refer to the figure below). Throughout this factorization process, the vacant entries in the ratings matrix can be populated, serving as predictions for user ratings on items. These predictions can then be utilized to provide recommendations to users.

<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/matrix-factorization.png">
<br>
<em> Matrix factorization in collaborative filtering</em></p>

[ALS (alternating least squares)](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-2-alternating-least-square-als-matrix-4a76c58714a1) is a mathematically optimized implementation of collaborative filtering that uses Alternating Least Squares (ALS) with Weighted-Lamda-Regularization (ALS-WR) to find optimal factor weights that minimize the least squares between predicted and actual ratings. [Spark's MLLib package](https://spark.apache.org/docs/latest/ml-guide.html) has a [built-in ALS function](https://spark.apache.org/docs/latest/mllib-collaborative-filtering.html), and we will use it in this post.

## 2. System setup
* Ubuntu 22.04.3
* JDK 11.0.21
* IntelliJ IDEA (2023.2.5)
* Scala 2.12.18
* Spark 3.5.0
* Hadoop 3.6.6

## 3. Dataset
In this project we will use the [MovieLens dataset](https://grouplens.org/datasets/movielens/). You can download **ml-100k** (6.19MB) by running:
```
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
```
Unzip the zip file by running:
```
unzip ml-100k.zip
```
I have added few more files and rename the folder to ml-900MB(just to increase the file size).You will see unzipped `ml-900MB` folder contain multiple files.
<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/ml-100k.png"></p>

We mainly use two data files:
* `u.data`: user ratings data, includes **user id**, **item id**, **rating**, **timestamp**.
* `u.item`: movies data, includes **item id**, **movie title**, **release date**, **imdb url**, etc.

## 4. Runnning in Spark

### 4.1 Clone code from Github
Before running in Spark, clone code from my [Github Repository](https://github.com/DharmeshTewari/CPSC-531_Movie-Recommendation-System.git) to your local directory using:
```
git clone https://github.com/DharmeshTewari/CPSC-531_Movie-Recommendation-System.git
```
Open the folder in IntelliJ IDEA. Your project structure should look like this:
<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/scala-project-structure.png"></p>

### 4.2 Preparing data in HDFS
Befofe we start, we need to start hadoop HDFS and YARN services in terminal
```
$ hadoop namenode -format
$ start-all.sh
```
Then we need to upload ml-900MB dataset to Hadoop HDFS:
```
$ hadoop fs -put ~/ml-900MB /user/dharmesh/movierecommendation
```

### 4.3 Train recommendataion model in Spark
Train a recommendation model in Spark using:
```
$ spark-submit --driver-memory 512m --executor-cores 2 --class MovieRecommenderTrainer --master yarn --deploy-mode client ~/MovieRecommendationSystem/out/artifacts/MovieRecommendationSystem_jar/MovieRecommendationSystem.jar
```
Check out your trained model in HDFS using:
```
$ hadoop fs -ls -h /user/dharmesh/movierecommendation
```
You will see your model here:
<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/spark-recommender-train.png"></p>

### 4.4 Generating recommendations in Spark
Recommend movies for `userID=150` in Spark using:
```
$ spark-submit --driver-memory 512m --executor-cores 2 --class MovieRecommender --master yarn --deploy-mode client ~/MovieRecommendationSystem/out/artifacts/MovieRecommendationSystem_jar2/MovieRecommendationSystem.jar --U 150
```
You will see this output:
<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/spark-recommend-movie.png"></p>

Or recommend users for `movieID=50` in Spark using:
```
$ spark-submit --driver-memory 512m --executor-cores 2 --class MovieRecommender --master yarn --deploy-mode client ~/MovieRecommendationSystem/out/artifacts/MovieRecommendationSystem_jar2/MovieRecommendationSystem.jar --M 50
```
You will see this output:
<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/spark-recommend-user.png"></p>

## 5. Movie Recommendation system design
Our system design are as below.
<p align="center">
<img src="/imgs/2023-12-04-build-recommendation-system-using-scala-spark-and-hadoop/recommendation-system-design.png"></p>

In summary, there are two Scala objects:
* `MovieRecommenderTrainer.scala`: reads ratings file (`u.data`), prepares data, trains ALS model and saves model.
* `MovieRecommender.scala`: reads movies file (`u.item`), loads ALS model, generating movie recommendations.

## 6. Implementation

### 6.1 Training ALS model - `MovieRecommenderTrainer.scala`
`MovieRecommenderTrainer.scala` is a Scala object that contains three main methods.
#### 6.1.1 `prepareData`
`prepareData` reads **ratings** data from path, parses useful fields and returns `ratingsRDD`.
```
  def prepareData(sc: SparkContext, dataPath: String): RDD[Rating] = {
    val file: RDD[String] = sc.textFile(dataPath)
    val ratingsRDD: RDD[Rating] = file.map(line => line.split("\t") match {
      case Array(user, item, rate, _) => Rating(user.toInt, item.toInt, rate.toDouble)
    })
    println("Sample Rating: " + ratingsRDD.first())
    ratingsRDD
  }
```
#### 6.1.2 `ALS.train`
`ALS.train` does explicit rating training of `ratingsRDD` and returns a `MatrixFactorizationModel` object.
```
  def trainALSModel(ratingsRDD: RDD[Rating]): MatrixFactorizationModel = {
    println("Start ALS training, rank=5, iteration=20, lambda=0.1")
    ALS.train(ratingsRDD, 5, 20, 0.1)
  }
```
Information about training parameters:

|Parameter | Description |
|--|-----|
|ratings|RDD with a format of Rating(userID, productID, rating)|
|rank|during matrix generation, the original matrix <em>A(m x n)</em> is decomposed into <em>X(m x **rank**)</em> and <em>Y(**rank** x n)</em>, in which <em>**rank**</em> essentially means the number of latent factors/features that you can specify.|
|iterations|number of ALS calculation iterations (default=5)|
|lambda|regularization factor (default=0.01)

#### 6.1.3 `saveModel`
`saveModel` saves model to path.
```
  def saveModel(context: SparkContext, model: MatrixFactorizationModel, modelPath: String): Unit = {
    try {
      model.save(context, modelPath)
    } catch {
      case _: Exception => println("Error occurred when saving the model.")
    }
}
```

### 6.2 Generating recommendations - `MovieRecommender.scala`
`MovieRecommender.scala` is a Scala object that contains four main methods.
#### 6.2.1 `prepareData`
`loadData` reads **movies** data from path, parses useful fields and returns `movieTitle`.
```
  def loadData(sc: SparkContext, dataPath: String): RDD[(Int, String)] = {
    println("Loading Data...")
    val itemRDD: RDD[String] = sc.textFile(dataPath)
    val movieTitle: RDD[(Int, String)] = itemRDD.map(line => line.split("\\|")).map(x => (x(0).toInt, x(1)))
    movieTitle
  }
```
#### 6.2.2 `MatrixFactorizationModel.load`
`MatrixFactorizationModel.load` loads ALS model from path.
```
val model: MatrixFactorizationModel = MatrixFactorizationModel.load(sc, modelPath)
```
#### 6.2.3 `model.recommendProducts`
`recommendMovies` recommends movies for given userID.
```
val recommendations = model.recommendProducts(inputUserID, 20)
```
#### 6.2.4 `model.recommendUsers`
`recommendUsers` recommends users for given itemID.
```
val recommendations = model.recommendUsers(inputMovieID, 20)
```

## 7. Summary
We have built a recommendataion system using Ubantu, Scala, Spark, and Hadoop. Thanks!!

## Contact
* **Author**: Dharmesh Tewari
* **Email**: [dtewari2711@gmail.com](dtewari2711@gmail.com)
* **Github**: [https://github.com/DharmeshTewari](https://github.com/DharmeshTewari)
* **Linkedin**: [www.linkedin.com/in/dharmesh-tewari-501818a8](www.linkedin.com/in/dharmesh-tewari-501818a8)

