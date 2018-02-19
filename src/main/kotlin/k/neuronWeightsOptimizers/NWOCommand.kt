package k.neuronWeightsOptimizers

import k.nets.NeuralNetwork
import k.neurons.AbstractMLPNeuron

interface NWOCommand {
    val showLogs: Boolean

    fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork)
}