package k.neurons

import k.utils.getEuclideanDistance
import java.util.*

class GaussianNeuron(val prevLayer: ArrayList<Neuron>, override var value: Double = 0.0) : Neuron {

    var center: DoubleArray = DoubleArray(prevLayer.size)
    var radius: Double = 1.0

    var dEdc: DoubleArray = DoubleArray(prevLayer.size)
    var dEdr: Double = 0.0

    override fun calculateState() {
        val x = DoubleArray(prevLayer.size)

        for (i in prevLayer.indices) {
            x[i] = prevLayer[i].value
        }

        val dist = getEuclideanDistance(x, center)

        value = Math.exp(-Math.pow(dist, 2.0) / (2 * Math.pow(radius, 2.0)))

        //value = 1 / (1 + Math.exp(-1 * Math.pow(dist, 2.0) / Math.pow(radius, 2.0)))
    }
}