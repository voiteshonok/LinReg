import au.com.bytecode.opencsv.CSVWriter
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.abs
import breeze.stats.mean

import java.io.{BufferedWriter, FileWriter}
import java.util.logging.Logger

case class LinearRegression(Logger: Logger){
  var w: DenseVector[Double] = DenseVector.fill(1)(0)
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
