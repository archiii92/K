package k.neurons

/* Нейрон с логистической функцией активации */
class LogisticNeuron(prevLayer: ArrayList<Neuron>) : AbstractNeuron(prevLayer) {

    var k: Double = 1.0 // Коэффициент логистической функции

    /* Логистическая функция */
    override fun activationFunction(x: Double): Double {
        return 1 / (1 + Math.exp(-x * k))
    }

    /* Производная логистической функции */
    override fun activationFunctionDerivative(x: Double): Double {
        val f = activationFunction(x)
        return f * (1 - f)
    }
}