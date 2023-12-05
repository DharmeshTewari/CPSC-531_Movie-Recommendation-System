// Define Scala version for the project
scalaVersion := "2.12.18"

// Project metadata
name := "MovieRecommendationSystem"
organization := "com.example.scala"
version := "1.0"

// Define library dependencies
libraryDependencies ++= Seq(
  "org.scala-lang.modules" %% "scala-parser-combinators" % "2.2.0",
  "org.apache.spark" %% "spark-core" % "3.5.0",
  "org.apache.spark" %% "spark-mllib" % "3.5.0",
  "org.apache.spark" %% "spark-sql" % "3.5.0",
  "org.apache.spark" %% "spark-hive" % "3.5.0"
)
