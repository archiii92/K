package kotlinFMLP.neurons

interface Neuron {
    var outputValue: Double
    fun calculateState()
}