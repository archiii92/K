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
    fun setInputValue(dataVector: DataVector)
    fun calculateOutput(dataVector: DataVector)
    fun calculateHiddenLayer()
    fun calculateOutputLayer()
    fun getOutputValue(): ArrayList<Double>
}