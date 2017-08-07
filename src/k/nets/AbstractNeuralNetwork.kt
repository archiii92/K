package k.nets

import k.neurons.Neuron
import k.utils.DataVector
import k.utils.denormalized
import k.utils.normalized

abstract class AbstractNeuralNetwork(
        final override val dataFileName: String,
        final override val trainTestDivide: Int,
        val inputLayerSize: Int,
        val outputLayerSize: Int
) : NeuralNetwork {
    final override val trainData: ArrayList<DataVector> = ArrayList()
    final override val testData: ArrayList<DataVector> = ArrayList()

    final override val inputLayer: ArrayList<Neuron> = ArrayList(inputLayerSize)
    final override val outputLayer: ArrayList<Neuron> = ArrayList(outputLayerSize)

    final override fun prepareData() {
        k.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
    }

    final override fun test() {
        var trainError: Double = 0.0
        var testError: Double = 0.0

        for (dataVector: DataVector in trainData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                trainError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println(String.format("%1.1f | %1.1f", dataVector.Forecast[i], result[i]))
            }
        }
        trainError = Math.sqrt(trainError / (trainData.size))

        for (dataVector: DataVector in testData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                testError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println(String.format("%1.1f | %1.1f", dataVector.Forecast[i], result[i]))
            }
        }
        testError = Math.sqrt(testError / (testData.size))

        System.out.println(String.format("Train %1.3f | Test %1.3f", trainError, testError))
    }

    final override fun setInputValue(dataVector: DataVector) {
        for (i in inputLayer.indices) {
            val normalizedValue = normalized(dataVector.Window[i])
            inputLayer[i].value = normalizedValue
        }
    }

    final override fun calculateOutputLayer() {
        for (neuron: Neuron in outputLayer) {
            neuron.calculateState()
        }
    }

    final override fun getOutputValue(): ArrayList<Double> {
        return ArrayList(outputLayer.map { x -> denormalized(x.value) })
    }
}