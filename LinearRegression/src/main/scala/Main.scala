import Utils._

object Main {
  def main(args: Array[String]): Unit = {
    val trainDataset = getDataset("/home/slava/Desktop/BigData-Made-2021/HW3/train.csv")
    val X_train = trainDataset._1
    val y_train = trainDataset._2

    val testDataset = getDataset("/home/slava/Desktop/BigData-Made-2021/HW3/test.csv")
    val X_test = testDataset._1
    val y_test = testDataset._2

    val logger = getLogger("/home/slava/Desktop/BigData-Made-2021/HW3/log.txt")

    val reg = LinearRegression(logger)
    reg.fit(X_train, y_train)

    reg.predict(X_test, "/home/slava/Desktop/BigData-Made-2021/HW3/out.csv")

    logger.info(f"loss = ${reg.loss(X_test, y_test)}")
  }
}
