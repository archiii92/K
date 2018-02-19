package kotlinFMLP.layers

import kotlinFMLP.neurons.Neuron

interface Layer {
    val neurons: ArrayList<Neuron>
    var inputVector: DoubleArray
    val outputVector: DoubleArray

    fun build()
    fun calculate()
    fun clear()
}