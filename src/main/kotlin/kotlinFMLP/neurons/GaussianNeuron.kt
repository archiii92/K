package kotlinFMLP.neurons

class GaussianNeuron(inputVectorSize: Int) : Neuron {
    var inputVector = DoubleArray(inputVectorSize)
    override var outputValue = 0.0

    var center = DoubleArray(inputVectorSize)
    var radius = DoubleArray(inputVectorSize)

    var δ = 0.0
    var Δc = DoubleArray(inputVectorSize)
    var Δr = DoubleArray(inputVectorSize)

    init {
        for (i in radius.indices) {
            radius[i] = 1.0
        }
    }

    override fun calculateState() {
        outputValue = Math.exp(-u() / 2)
    }

    fun u(): Double {
        var sum = 0.0
        for (i in center.indices) {
            val numerator = Math.pow(inputVector[i] - center[i], 2.0)
            val denominator = Math.pow(radius[i], 2.0)
            sum += numerator / denominator
        }
        return sum
    }
}