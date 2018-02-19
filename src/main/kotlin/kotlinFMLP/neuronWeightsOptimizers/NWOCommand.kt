package kotlinFMLP.neuronWeightsOptimizers

import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neurons.AbstractMLPNeuron

interface NWOCommand {
    val showLogs: Boolean

    fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork)
}