import breeze.linalg._
import breeze.stats.mean
import breeze.numerics.abs

import java.io._

import java.io.{BufferedWriter, FileWriter}
import au.com.bytecode.opencsv.CSVWriter

import java.util.logging.{FileHandler, Logger, SimpleFormatter}

val file = "/home/slava/Desktop/BigData-Made-2021/HW3/train.csv"
val data: DenseMatrix[Double] = csvread(new File(file), separator = ',', skipLines = 1)

print(data.cols, data.rows)

val y = data(::, data.cols - 1)

val X = data(::, 0 to data.cols - 2)

val logger = Logger.getLogger("name")
val handler = new FileHandler("/home/slava/Desktop/BigData-Made-2021/HW3/log.txt")
val formatter = new SimpleFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

class LinearRegression(Logger: Logger){
  var w: DenseVector[Double] = DenseVector.fill(1)(0.2)
  var logger: Logger = Logger

  def fit(X: DenseMatrix[Double], y: DenseVector[Double], iters: Int = 100, lr_base: Double = 0.01): Unit = {
    val ones = DenseMatrix.fill[Double](X.rows, 1)(1)
    val ones_X = DenseMatrix.horzcat(ones, X)
    w = DenseVector.fill(ones_X.cols)(0)

    for (epoch <- 0 until iters) {
      for (i <- 0 until ones_X.rows) {
        val grad = ones_X(i, ::) * (ones_X(i, ::) * w - y(i))
        val lr = lr_base / ((i + 10) / 10)
        w = w - lr * grad.t
      }
      val mae = mean(abs(y - ones_X * w))
      logger.info(f"Epoch: $epoch, MAE=$mae")
    }
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    val ones = DenseMatrix.fill[Double](X.rows, 1)(1)
    val ones_X = DenseMatrix.horzcat(ones, X)
    ones_X * w
  }

  def predict(X: DenseMatrix[Double], path: String): Unit = {
    val ones = DenseMatrix.fill[Double](X.rows, 1)(1)
    val ones_X = DenseMatrix.horzcat(ones, X)
    val result = ones_X * w

    val out = new BufferedWriter(new FileWriter(path))
    val writer = new CSVWriter(out)

    writer.writeNext(Array(result.toString()))
    out.close()
  }

  def loss(X: DenseMatrix[Double], y_true: DenseVector[Double]): Double = {
    mean(abs(y_true - predict(X)))
  }
}

val reg = new LinearRegression(logger)
reg.fit(X, y)

val file = "/home/slava/Desktop/BigData-Made-2021/HW3/test.csv"
val data: DenseMatrix[Double] = csvread(new File(file), separator = ',', skipLines = 1)

print(data.cols, data.rows)

val y = data(::, data.cols - 1)

val X = data(::, 0 to data.cols - 2)
println(reg.predict(X))
println(reg.loss(X, y))

//reg.predict(X, "/home/slava/Desktop/BigData-Made-2021/HW3/out.csv")
