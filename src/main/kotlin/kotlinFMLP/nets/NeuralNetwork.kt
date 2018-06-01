package kotlinFMLP.nets

import kotlinFMLP.neuronWeightsOptimizers.NWOCommand
import kotlinFMLP.utils.DataVector

interface NeuralNetwork {
    val dataFileName: String
    val trainTestDivide: Int

    val trainData: ArrayList<DataVector>
    val testData: ArrayList<DataVector>

    val showLogs: Boolean

    fun prepareData()
    fun buildNetwork()
    fun learn()
    fun calculate(dataVector: DataVector): DoubleArray
    fun optimizeMLPNeuronWeigths(neuronWeightsOptimizer: NWOCommand)
    fun calculateError(data: ArrayList<DataVector>): Double
    fun clearNetwork()
    fun shuffleData()
}