package k.neurons

import java.util.*

abstract class AbstractMLPNeuron(inputVectorSize: Int) : Neuron {
    var inputVector = DoubleArray(inputVectorSize)
    final override var outputValue = 0.0

    var sum = 0.0
    val weights = DoubleArray(inputVectorSize)

    var δ = 0.0
    var ΔW = DoubleArray(inputVectorSize)

    init {
        val r = Random()
        for (i in weights.indices) {
            weights[i] = r.nextDouble()
        }
    }

    override fun calculateState() {
        sum = inputVector.indices.sumByDouble {
            inputVector[it] * weights[it]
        }
        outputValue = activationFunction(sum)
    }

    abstract fun activationFunction(x: Double): Double

    abstract fun activationFunctionDerivative(x: Double): Double
}