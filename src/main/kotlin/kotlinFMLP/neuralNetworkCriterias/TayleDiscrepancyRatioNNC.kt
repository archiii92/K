package kotlinFMLP.neuralNetworkCriterias

import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.utils.DataVector

class TayleDiscrepancyRatioNNC : NNCCommand {
    override fun calculateCriteria(data: ArrayList<DataVector>, neuralNetwork: NeuralNetwork): Double {

        var numerator = 0.0
        var forecastSquareSum = 0.0
        var realSquareSum = 0.0
        for (dataVector in data) {
            val result = neuralNetwork.calculate(dataVector)
            for (i in dataVector.Forecast.indices) {
                numerator += Math.pow(result[i] - dataVector.Forecast[i], 2.0)

                forecastSquareSum += Math.pow(result[i], 2.0)
                realSquareSum += Math.pow(dataVector.Forecast[i], 2.0)
            }
        }

        val denominator = Math.sqrt(forecastSquareSum) + Math.sqrt(realSquareSum)

        return Math.sqrt(numerator) / denominator
    }
}