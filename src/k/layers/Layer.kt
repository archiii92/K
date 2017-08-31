package k.layers

import k.neurons.Neuron

interface Layer {
    val neurons: ArrayList<Neuron>
    var inputVector: DoubleArray
    val outputVector: DoubleArray

    fun build()
    fun calculate()
}