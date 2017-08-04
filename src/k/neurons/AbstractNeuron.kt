package k.neurons

import java.util.*

abstract class AbstractNeuron(prevLayer: ArrayList<Neuron>) : Neuron {
    final override var prevLayer: ArrayList<Neuron> = prevLayer
    final override var weights: ArrayList<Double> = ArrayList(prevLayer.size)

    override var value: Double = 0.0
    override var sum: Double = 0.0

    override var δ: Double = 0.0
    override var ΔW: ArrayList<Double> = ArrayList<Double>(prevLayer.size)

    init {
        val r: Random = Random()
        for (i in prevLayer.indices) {
            weights.add(r.nextDouble())
        }
    }

    final override fun calculateState() {
        sum = 0.0
        for (i in prevLayer.indices) {
            sum += prevLayer[i].value * weights[i]
        }
        value = activationFunction(sum)
    }

    override abstract fun activationFunction(x: Double): Double

    override abstract fun activationFunctionDerivative(x: Double): Double
}