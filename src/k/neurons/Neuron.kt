package k.neurons

abstract class Neuron {
    var value: Double = 0.0

    open fun calculateState(): Double {
        return 0.0
    }
}