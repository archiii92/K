package k.neurons

import java.util.*

/* Нейрон с логистической функцией активации */
class LogisticNeuron : Neuron {
    constructor(inputSize: Int, outputSize: Int) {
        val inputNeurons: ArrayList<Neuron> = ArrayList(inputSize)
        val weights: ArrayList<Double> = ArrayList(inputSize)
    }

    /* Логистическая функция */
    fun activationFunction(x: Double): Double {
        return 1 / 1 + Math.pow(Math.E, -x)
    }

    /* Производная логистической функции */
    fun activationFunctionDerivative(x: Double): Double {
        val v = activationFunction(x)
        return v * (1 - v)
    }
}