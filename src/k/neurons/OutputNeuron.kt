package k.neurons

import java.util.*

class OutputNeuron : Neuron {
    constructor(inputSize: Int) {
        val inputValues: ArrayList<Double> = ArrayList(inputSize)
        val outputValue: Double = 0.0
    }
}