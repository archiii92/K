package k.neuronWeightsInitializerCommands

import k.neurons.AbstractMLPNeuron

interface NWICommand {
    fun initializeWeights(neuron: AbstractMLPNeuron)
}