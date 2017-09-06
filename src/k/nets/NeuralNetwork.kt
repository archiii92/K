package k.nets

import k.neuronWeightsOptimizers.NWOCommand
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

    fun optimizeMLPNeuronWeigths()
    fun calculateError(data: ArrayList<DataVector>): Double
}