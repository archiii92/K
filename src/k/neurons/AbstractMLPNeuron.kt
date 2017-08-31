package k.neurons

import java.util.*

abstract class AbstractMLPNeuron(inputVectorSize: Int) : Neuron {
    var inputVector: DoubleArray = DoubleArray(inputVectorSize)
    final override var outputValue: Double = 0.0

    var sum: Double = 0.0
    val weights: DoubleArray = DoubleArray(inputVectorSize)

    var δ: Double = 0.0
    var ΔW: DoubleArray = DoubleArray(inputVectorSize)

    init {
        val r = Random()
        for (i in inputVector.indices) {
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