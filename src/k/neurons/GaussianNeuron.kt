package k.neurons

import k.utils.getEuclideanDistance

class GaussianNeuron(val prevLayer: ArrayList<Neuron>, override var value: Double = 0.0) : Neuron {

    val center: ArrayList<Double> = ArrayList(prevLayer.size)
    var radius: Double = 1.0

//    init {
//        for(i in center.indices){
//            center[i] = 0.0
//        }
//    }

    override fun calculateState() {
        val x: ArrayList<Double> = ArrayList(prevLayer.size)

        for (i in prevLayer.indices) {
            x.add(prevLayer[i].value)
        }

        val dist = getEuclideanDistance(x, center)

        //value = Math.exp(-1 * Math.pow(dist, 2.0) / 2 * Math.pow(radius, 2.0))

        value = Math.exp(-1 * Math.pow(dist / radius, 2.0))
    }
}