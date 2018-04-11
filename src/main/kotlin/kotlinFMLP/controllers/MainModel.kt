package kotlinFMLP.controllers

import kotlinFMLP.nets.FMLP
import kotlinFMLP.nets.IFMLP
import kotlinFMLP.nets.MLP
import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neuralNetworkCriterias.NNCCommand
import kotlinFMLP.neuralNetworkCriterias.TayleDiscrepancyRatioNNC
import kotlinFMLP.neuronFactories.AbstractNeuronFactory
import kotlinFMLP.neuronFactories.LogisticNeuronFactory
import kotlinFMLP.neuronWeightsOptimizers.NWOCommand
import kotlinFMLP.neuronWeightsOptimizers.ParticleSwarmNWO
import kotlinFMLP.neuronWeightsOptimizers.SimulatedAnnealingNWO

class MainModel {
    fun makeForecast(forecastSettings: ForecastSettings): ForecastResult {
        val neuronFactory: AbstractNeuronFactory = LogisticNeuronFactory()
        val neuralNetworkCriteria: NNCCommand = TayleDiscrepancyRatioNNC()
        var neuronWeightsOptimizer: NWOCommand? = null
        var neuralNetwork: NeuralNetwork? = null

        when (forecastSettings.selectedAlgorithm) {
            "sa" -> neuronWeightsOptimizer = SimulatedAnnealingNWO(
                    forecastSettings.initialTemperature.toDouble(),
                    forecastSettings.warmingKeepPercent.toDouble() / 100
            )
            "pso" -> neuronWeightsOptimizer = ParticleSwarmNWO(
                    forecastSettings.particleCount.toInt(),
                    forecastSettings.iterationCount.toInt(),
                    forecastSettings.φp.toDouble(),
                    forecastSettings.φg.toDouble(),
                    forecastSettings.k.toDouble()
            )
        }

        when (forecastSettings.selectedNetwork) {
            "mpl" -> neuralNetwork = MLP(
                    forecastSettings.fileName,
                    forecastSettings.trainTestDivide.toInt(),
                    forecastSettings.inputLayerSize.toInt(),
                    forecastSettings.hiddenLayerSize.toInt(),
                    1,
                    0.001,
                    0.000001,
                    10000,
                    neuronFactory,
                    neuralNetworkCriteria
            )
            "fmlp" -> neuralNetwork = FMLP(
                    forecastSettings.fileName,
                    forecastSettings.trainTestDivide.toInt(),
                    forecastSettings.inputLayerSize.toInt(),
                    forecastSettings.fuzzyLayerSize.toInt(),
                    forecastSettings.hiddenLayerSize.toInt(),
                    1,
                    0.0000001,
                    100,
                    0.001,
                    0.00001,
                    5000,
                    3,
                    neuronFactory,
                    neuralNetworkCriteria
            )
        }

        val result = ForecastResult()

        if (neuralNetwork != null && neuronWeightsOptimizer != null) {

            neuralNetwork.prepareData()
            neuralNetwork.buildNetwork()

            val initError = neuralNetwork.calculateError(neuralNetwork.testData)

            if (neuralNetwork is IFMLP) {
                neuralNetwork.initFuzzyLayer()
                val afterFuzzyLayerInitError = neuralNetwork.calculateError(neuralNetwork.testData)
                result.afterFuzzyLayerInitError = afterFuzzyLayerInitError
            }

            neuralNetwork.optimizeMLPNeuronWeigths(neuronWeightsOptimizer)
            val afterOptimizationError = neuralNetwork.calculateError(neuralNetwork.testData)

            neuralNetwork.learn()
            val finalError = neuralNetwork.calculateError(neuralNetwork.testData)

            result.initError = initError
            result.afterOptimizationError = afterOptimizationError
            result.finalError = finalError

            result.realValues = DoubleArray(neuralNetwork.testData.size)
            result.forecastValues = DoubleArray(neuralNetwork.testData.size)

            for (i in neuralNetwork.testData.indices) {
                val dataVector = neuralNetwork.testData[i]
                val res = neuralNetwork.calculate(dataVector)
                for (j in dataVector.Forecast.indices) {
                    result.realValues[i] = dataVector.Forecast[j]
                    result.forecastValues[i] = res[j]
                }
            }

            neuralNetwork.clearNetwork()
        }

        return result
    }
}