package k.layers

import k.neurons.GaussianNeuron
import k.neurons.Neuron
import k.neurons.SingleNeuron

class FuzzyLayer(layerSize: Int, val inputVectorSize: Int) : AbstractLayer(layerSize) {
    override val neurons: ArrayList<Neuron> = ArrayList(layerSize + 1)

    override fun build() {
        var i = 0
        while (i < layerSize) {
            neurons.add(GaussianNeuron(inputVectorSize))
            i++
        }
        neurons.add(SingleNeuron())
    }

    override var inputVector: DoubleArray = DoubleArray(inputVectorSize)
        set(value) {
            var i = 0
            while (i < layerSize) {
                val gaussNeuron = neurons[i] as GaussianNeuron
                gaussNeuron.inputVector = value
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