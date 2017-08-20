package k.neurons

import k.utils.getEuclideanDistance
import java.util.*

class GaussianNeuron(val prevLayer: ArrayList<Neuron>, override var value: Double = 0.0) : Neuron {

    var center: DoubleArray = DoubleArray(prevLayer.size)
    var radius: Double = 1.0
    val weights: DoubleArray = DoubleArray(prevLayer.size)

    var δ: Double = 0.0
    var ΔW: DoubleArray = DoubleArray(prevLayer.size)

    init {
        val r: Random = Random()
        for (i in prevLayer.indices) {
            weights[i] = r.nextDouble()
        }
    }


    override fun calculateState() {
        val x: DoubleArray = DoubleArray(prevLayer.size)

        for (i in prevLayer.indices) {
            x[i] = prevLayer[i].value
        }

        val dist = getEuclideanDistance(x, center)

        value = Math.exp(-Math.pow(dist, 2.0) / (2 * Math.pow(radius, 2.0)))

        //value = 1 / (1 + Math.exp(-1 * Math.pow(dist, 2.0) / Math.pow(radius, 2.0)))
    }
}