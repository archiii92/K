package k.neuronFactories

import k.neurons.LogisticNeuron
import k.neurons.Neuron

class LogisticNeuronFactory : AbstractNeuronFactory() {
    override fun createNeuron(inputVectorSize: Int): Neuron {
        return LogisticNeuron(inputVectorSize)
    }
}