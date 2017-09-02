package k.neuronWeightsInitializerCommands

import k.neurons.AbstractMLPNeuron
import java.util.*

class RandomNWI : NWICommand {
    override fun initializeWeights(neuron: AbstractMLPNeuron) {
        val r = Random()
        for (i in neuron.weights.indices) {
            neuron.weights[i] = r.nextDouble()
        }
    }
}