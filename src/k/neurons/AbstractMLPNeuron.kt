package k.neurons

import k.neuronWeightsInitializerCommands.NWICommand

abstract class AbstractMLPNeuron(inputVectorSize: Int, NWICommand: NWICommand) : Neuron {
    var inputVector: DoubleArray = DoubleArray(inputVectorSize)
    final override var outputValue: Double = 0.0

    var sum: Double = 0.0
    val weights: DoubleArray = DoubleArray(inputVectorSize)

    var δ: Double = 0.0
    var ΔW: DoubleArray = DoubleArray(inputVectorSize)

    init {
        NWICommand.initializeWeights(this)
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