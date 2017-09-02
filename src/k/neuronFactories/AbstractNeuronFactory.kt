package k.neuronFactories

import k.neurons.Neuron

interface AbstractNeuronFactory {
    fun createNeuron(inputVectorSize: Int): Neuron
}