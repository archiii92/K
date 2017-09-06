package k.neuronWeightsOptimizers

import k.nets.NeuralNetwork
import k.neurons.AbstractMLPNeuron

interface NWOCommand {
    fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork)
}