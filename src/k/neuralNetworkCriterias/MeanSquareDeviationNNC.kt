package k.neuralNetworkCriterias

import k.nets.NeuralNetwork
import k.utils.DataVector

class MeanSquareDeviationNNC : NNCCommand {
    override fun calculateCriteria(data: ArrayList<DataVector>, neuralNetwork: NeuralNetwork): Double {
        var error = 0.0

        var sum = 0.0
        data.forEach { sum += it.Forecast[0] }
        val mean = sum / data.size

        for (dataVector in data) {
            val result = neuralNetwork.calculate(dataVector)
            for (i in dataVector.Forecast.indices) {
                error += Math.pow(result[i] - mean, 2.0)
            }
        }

        return Math.sqrt(error / (data.size - 1))
    }
}