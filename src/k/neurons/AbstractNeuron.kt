package k.neurons

import java.util.*

abstract class AbstractNeuron(prevLayer: ArrayList<Neuron>) : Neuron {
    final override val prevLayer: ArrayList<Neuron> = prevLayer
    final override val weights: ArrayList<Double> = ArrayList(prevLayer.size)

    final override var value: Double = 0.0
    final override var sum: Double = 0.0

    final override var δ: Double = 0.0
    final override var ΔW: ArrayList<Double> = ArrayList<Double>(prevLayer.size)

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
}