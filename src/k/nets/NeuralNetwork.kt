package k.nets

import k.utils.DataVector

interface NeuralNetwork {
    val dataFileName: String
    val trainTestDivide: Int

    val trainData: ArrayList<DataVector>
    val testData: ArrayList<DataVector>

    fun prepareData()
    fun buildNetwork()
    fun learn()
    fun test()
    fun calculate(dataVector: DataVector): DoubleArray
}