package k.neurons

/* Нейрон с функцией активации в виде гиперболического тангенса */
class HyperbolicTangentNeuron(prevLayer: ArrayList<Neuron>) : AbstractNeuron(prevLayer) {

    /* Гиперболический тангенс */
    override fun activationFunction(x: Double): Double {
        val a = Math.exp(x)
        val b = Math.exp(-x)
        return (a - b) / (a + b)
    }

    /* Производная гиперболического тангенса */
    override fun activationFunctionDerivative(x: Double): Double {
        val f = activationFunction(x)
        return 1 - Math.pow(f, 2.0)
    }
}