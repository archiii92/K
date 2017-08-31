package k.neurons

import k.utils.getEuclideanDistance

class GaussianNeuron(val inputVectorSize: Int) : Neuron {

    var inputVector: DoubleArray = DoubleArray(inputVectorSize)
    override var outputValue: Double = 0.0

    var center: DoubleArray = DoubleArray(inputVectorSize)
    var radius: Double = 1.0

    var dEdc: DoubleArray = DoubleArray(inputVectorSize)
    var dEdr: Double = 0.0

    override fun calculateState() {
//        val x = DoubleArray(inputVectorSize)
//
//        for (i in prevLayer.indices) {
//            x[i] = prevLayer[i].outputValue
//        }

        outputValue = GaussFunction(getEuclideanDistance(inputVector, center))
    }

    fun GaussFunction(dist: Double): Double {

        //return 1 / (1 + Math.exp(-Math.pow(dist, 2.0) / Math.pow(radius, 2.0)))
        return Math.exp(-Math.pow(dist, 2.0) / (2 * Math.pow(radius, 2.0)))
    }
}