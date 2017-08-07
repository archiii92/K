package k.nets

import k.neurons.Neuron
import k.utils.DataVector

interface NeuralNetwork {
    val dataFileName: String
    val trainTestDivide: Int

    val trainData: ArrayList<DataVector>
    val testData: ArrayList<DataVector>

    val inputLayer: ArrayList<Neuron>
    val outputLayer: ArrayList<Neuron>

    fun prepareData()
    fun buildNetwork()
    fun learn()
    fun test()
    fun calculateOutput(dataVector: DataVector)
    fun setInputValue(dataVector: DataVector)
    fun calculateHiddenLayer()
    fun calculateOutputLayer()
    fun getOutputValue(): ArrayList<Double>
}