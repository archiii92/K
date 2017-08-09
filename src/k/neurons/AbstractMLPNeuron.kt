package k.neurons

import java.util.*

abstract class AbstractMLPNeuron(prevLayer: ArrayList<out Neuron>) : Neuron {
    final override var value: Double = 0.0

    var sum: Double = 0.0

    val prevLayer: ArrayList<out Neuron> = prevLayer
    val weights: ArrayList<Double> = ArrayList(prevLayer.size)

    var δ: Double = 0.0
    var ΔW: ArrayList<Double> = ArrayList<Double>(prevLayer.size)

    init {
        val r: Random = Random()
        for (i in prevLayer.indices) {
            weights.add(r.nextDouble())
        }
    }

    override fun calculateState() {
        sum = prevLayer.indices.sumByDouble { prevLayer[it].value * weights[it] }
        value = activationFunction(sum)
    }

    abstract fun activationFunction(x: Double): Double

    abstract fun activationFunctionDerivative(x: Double): Double
}