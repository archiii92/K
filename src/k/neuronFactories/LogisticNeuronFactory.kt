package k.neuronFactories

import k.neuronWeightsInitializerCommands.NWICommand
import k.neurons.LogisticNeuron
import k.neurons.Neuron

class LogisticNeuronFactory(val NWICommand: NWICommand) : AbstractNeuronFactory {
    override fun createNeuron(inputVectorSize: Int): Neuron {
        return LogisticNeuron(inputVectorSize, NWICommand)
    }
}