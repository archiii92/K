package k.layers

import k.neuronFactories.AbstractNeuronFactory
import k.neurons.AbstractMLPNeuron
import k.neurons.Neuron
import k.utils.denormalize

class OutputLayer(layerSize: Int, val inputVectorSize: Int, val neuronFactory: AbstractNeuronFactory) : AbstaractLayer(layerSize) {
    override val neurons: ArrayList<Neuron> = ArrayList(layerSize)

    override fun build() {
        var i = 0
        while (i < layerSize) {
            neurons.add(neuronFactory.createNeuron(inputVectorSize + 1))
            i++
        }
    }

    override var inputVector: DoubleArray = DoubleArray(inputVectorSize + 1)
        set(value) {
            var i = 0
            while (i < layerSize) {
                val hiddenNeuron = neurons[i] as AbstractMLPNeuron
                hiddenNeuron.inputVector = value
                i++
            }
            field = value
        }

    override var outputVector: DoubleArray = DoubleArray(layerSize)
        get() {
            var i = 0
            while (i < layerSize) {
                field[i] = denormalize(neurons[i].outputValue)
                i++
            }
            return field
        }
}