package k.layers

abstract class AbstaractLayer(val layerSize: Int) : Layer {
    final override fun calculate() {
        var i = 0
        while (i < layerSize) {
            neurons[i].calculateState()
            i++
        }
    }
}