package kotlinFMLP.neuronFactories

import kotlinFMLP.neurons.Neuron

interface AbstractNeuronFactory {
    fun createNeuron(inputVectorSize: Int): Neuron
}