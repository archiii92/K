package k.neurons

class InputNeuron : AbstractNeuron(ArrayList(0)) {
    override fun activationFunction(x: Double): Double {
        return 1.0
    }

    override fun activationFunctionDerivative(x: Double): Double {
        return 1.0
    }
}