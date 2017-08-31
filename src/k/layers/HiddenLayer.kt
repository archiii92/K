package k.layers

import k.neurons.AbstractMLPNeuron
import k.neurons.LogisticNeuron
import k.neurons.Neuron
import k.neurons.SingleNeuron

class HiddenLayer(layerSize: Int, val inputVectorSize: Int) : AbstaractLayer(layerSize) {
    override val neurons: ArrayList<Neuron> = ArrayList(layerSize + 1)

    override fun build() {
        var i = 0
        while (i < layerSize) {
            neurons.add(LogisticNeuron(inputVectorSize + 1))
            i++
        }
        neurons.add(SingleNeuron())
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