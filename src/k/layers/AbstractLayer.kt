package k.layers

abstract class AbstractLayer(val layerSize: Int) : Layer {
    final override fun calculate() {
        neurons.forEach { it.calculateState() }
    }


}