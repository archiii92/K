package k.neuronWeightsOptimizers

import k.nets.NeuralNetwork
import k.neurons.AbstractMLPNeuron
import k.utils.format
import k.utils.toFormatString
import java.util.*

class SimulatedAnnealingNWO(var T: Int) : NWOCommand {
    override fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
        val weightsChanges = DoubleArray(neuron.weights.size)
        val beforeError = neuralNetwork.calculateError(neuralNetwork.trainData)
        for (i in neuron.weights.indices) {
            val r = Random()
            val beforeWeight = neuron.weights[i]
            while (T > 0) {
                val currentError = neuralNetwork.calculateError(neuralNetwork.trainData)
                val oldWeight = neuron.weights[i]
                neuron.weights[i] = r.nextDouble()
                val newError = neuralNetwork.calculateError(neuralNetwork.trainData)

                val d = newError - currentError
                if (d > 0) {
                    val R = r.nextDouble()
                    val exp = Math.exp(-d / T)

                    if (exp <= R) neuron.weights[i] = oldWeight
                }
                T--
            }
            val afterWeigth = neuron.weights[i]

            weightsChanges[i] = beforeWeight - afterWeigth
        }
        val afterError = neuralNetwork.calculateError(neuralNetwork.trainData)
        println("Итог изм весов: ${weightsChanges.toFormatString(4)} Итог улуч ошибки: ${(beforeError - afterError).format(4)}")
    }
}