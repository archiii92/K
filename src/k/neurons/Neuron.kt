package k.neurons

interface Neuron {
    val prevLayer: ArrayList<Neuron>
    val weights: ArrayList<Double>

    var value: Double
    var sum: Double

    var δ: Double
    var ΔW: ArrayList<Double>

    fun calculateState()

    fun activationFunction(x: Double): Double

    fun activationFunctionDerivative(x: Double): Double
}