package k.neuronFactories

import k.neurons.Neuron

abstract class AbstractNeuronFactory {
    abstract fun createNeuron(inputVectorSize: Int): Neuron
}