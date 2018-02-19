package kotlinFMLP.neuronWeightsOptimizers

import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neurons.AbstractMLPNeuron
import kotlinFMLP.utils.format
import kotlinFMLP.utils.toFormatString
import java.util.*

class SimulatedAnnealingNWO(val t: Double, val r: Double, override val showLogs: Boolean = false) : NWOCommand {
    override fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
        val weightsChanges = DoubleArray(neuron.weights.size)
        val beforeError = neuralNetwork.calculateError(neuralNetwork.trainData)
        for (i in neuron.weights.indices) {
            val rand = Random()
            val beforeWeight = neuron.weights[i]
            var T = t
            while (T > 0) {
                val currentError = neuralNetwork.calculateError(neuralNetwork.trainData)
                val oldWeight = neuron.weights[i]
                neuron.weights[i] = rand.nextDouble()
                val newError = neuralNetwork.calculateError(neuralNetwork.trainData)

                val d = newError - currentError
                if (d > 0) {
                    val R = rand.nextDouble()
                    val exp = Math.exp(-d / T)

                    if (exp <= R) neuron.weights[i] = oldWeight
                }
                T -= T * r
            }
            val afterWeigth = neuron.weights[i]

            weightsChanges[i] = afterWeigth - beforeWeight
        }
        val afterError = neuralNetwork.calculateError(neuralNetwork.trainData)
        if (showLogs) println("Итог изм весов: ${weightsChanges.toFormatString(4)} Итог улуч ошибки: ${(beforeError - afterError).format(4)}")
    }

    override fun toString(): String {
        return "Алгоритм имитации отжига Q = ${t} и r = ${r}"
    }
}