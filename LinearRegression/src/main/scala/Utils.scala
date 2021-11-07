import breeze.linalg.{DenseMatrix, DenseVector, csvread}

import java.io.File
import java.util.logging.{FileHandler, Logger, SimpleFormatter}

object Utils {
  def getLogger(path: String): Logger ={
    val logger = Logger.getLogger("name")
    val handler = new FileHandler(path)
    val formatter = new SimpleFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger
  }

  def getDataset(path: String): (DenseMatrix[Double], DenseVector[Double]) ={
    val data: DenseMatrix[Double] = csvread(new File(path), separator = ',', skipLines = 1)

    val X = data(::, 0 to data.cols - 2)
    val y = data(::, data.cols - 1)
    (X, y)
  }
}
