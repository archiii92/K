package kotlinFMLP.neuronFactories

import kotlinFMLP.neurons.LogisticNeuron
import kotlinFMLP.neurons.Neuron

class LogisticNeuronFactory : AbstractNeuronFactory {
    override fun createNeuron(inputVectorSize: Int): Neuron {
        return LogisticNeuron(inputVectorSize)
    }
}