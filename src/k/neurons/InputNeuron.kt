package k.neurons

import k.utils.normalize

class InputNeuron(override var outputValue: Double = 0.0) : Neuron {
    override fun calculateState() {
        outputValue = normalize(outputValue)
    }
}