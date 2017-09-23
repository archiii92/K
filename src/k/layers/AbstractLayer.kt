package k.layers

import k.neurons.AbstractMLPNeuron
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
            }
        }
    }
}