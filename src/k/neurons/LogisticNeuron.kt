package k.neurons

import java.util.*
import kotlin.collections.ArrayList

/* Нейрон с логистической функцией активации */
class LogisticNeuron(prevLayer: ArrayList<Neuron>) : Neuron {
    override var prevLayer: ArrayList<Neuron> = prevLayer
    override var weights: ArrayList<Double> = ArrayList(prevLayer.size)

    override var value: Double = 0.0
    override var sum: Double = 0.0

    override var δ: Double = 0.0
    override var ΔW: ArrayList<Double> = ArrayList(prevLayer.size)

    var k = 1
    init {
        val r: Random = Random()
        for (i in prevLayer.indices) {
            weights.add(r.nextDouble())
        }
    }

    override fun calculateState() {
        sum = 0.0
        for (i in prevLayer.indices) {
            sum += prevLayer[i].value * weights[i]
        }
        value = activationFunction(sum)
    }

    /* Логистическая функция */
    private fun activationFunction(x: Double): Double {
        return 1 / 1 + Math.exp(-x * k)
    }

    /* Производная логистической функции */
    override fun activationFunctionDerivative(x: Double): Double {
        val v = activationFunction(x)
        return v * (1 - v)
    }
}