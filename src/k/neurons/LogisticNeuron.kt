package k.neurons

import java.util.*

/* Нейрон с логистической функцией активации */
class LogisticNeuron(inputSize: Int) : Neuron() {
    val inputNeurons: ArrayList<Neuron> = ArrayList(inputSize)
    var weights: ArrayList<Double> = ArrayList(inputSize)

    init {
        val r: Random = Random()
        for (i in weights.indices) {
            weights[i] = r.nextDouble()
        }
    }

    override fun calculateState(): Double {
        var sum: Double = 0.0
        for (i in inputNeurons.indices) {
            sum += inputNeurons[i].value * weights[i]
        }
        return activationFunction(sum)
    }

    /* Логистическая функция */
    private fun activationFunction(x: Double): Double {
        return 1 / 1 + Math.pow(Math.E, -x)
    }

    /* Производная логистической функции */
    fun activationFunctionDerivative(x: Double): Double {
        val v = activationFunction(x)
        return v * (1 - v)
    }
}