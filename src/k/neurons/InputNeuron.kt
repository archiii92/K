package k.neurons

class InputNeuron(
        override var prevLayer: ArrayList<Neuron> = ArrayList(0),
        override var weights: ArrayList<Double> = ArrayList(0),

        override var value: Double = 0.0,
        override var sum: Double = 0.0,

        override var δ: Double = 0.0,
        override var ΔW: ArrayList<Double> = ArrayList(0)
) : Neuron {
    override fun activationFunction(x: Double): Double {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun calculateState() {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun activationFunctionDerivative(x: Double): Double {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

}