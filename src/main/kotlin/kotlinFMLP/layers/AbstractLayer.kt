package kotlinFMLP.layers

import kotlinFMLP.neurons.AbstractMLPNeuron
import kotlinFMLP.neurons.GaussianNeuron
import java.util.*

abstract class AbstractLayer(val layerSize: Int) : Layer {
    final override fun calculate() {
        neurons.forEach { it.calculateState() }
    }
    final override fun clear(){
        val r = Random()
        neurons.forEach {
            if (it is AbstractMLPNeuron){
                for (i in it.weights.indices) {
                    it.weights[i] = r.nextDouble()
                }
            } else if (it is GaussianNeuron) {
                for (i in it.radius.indices) {
                    it.radius[i] = 1.0
                }
            }
        }
    }
}