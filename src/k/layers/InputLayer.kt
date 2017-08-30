package k.layers

import k.neurons.InputNeuron
import k.neurons.Neuron
import k.neurons.SingleNeuron

class InputLayer(override val layerSize: Int) : Layer {
    override val neurons: ArrayList<Neuron> = ArrayList(layerSize + 1)

    override fun build() {
        var i = 0
        while (i < layerSize) {
            neurons.add(InputNeuron())
            i++
        }
        neurons.add(SingleNeuron())
    }

    override var inputVector: DoubleArray = DoubleArray(layerSize)
        set(value) {
            var i = 0
            while (i < layerSize) {
                neurons[i].outputValue = value[i]
                i++
            }
        }

    override fun calculate() {
        var i = 0
        while (i < layerSize) {
            neurons[i].calculateState()
            i++
        }
    }

    override var outputVector: DoubleArray = DoubleArray(layerSize + 1)
        get() {
            var i = 0
            while (i < layerSize + 1) {
                field[i] = neurons[i].outputValue
                i++
            }
            return field
        }
}