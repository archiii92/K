package k.nets

import k.utils.DataVector

class FMLP(dataFileName: String,
           trainTestDivide: Int,
           inputLayerSize: Int,
           outputLayerSize: Int,
           hiddenLayerSize: Int,
           η: Double,
           errorThreshold: Double,
           iterationThreshold: Int
) : MLP(dataFileName, trainTestDivide, inputLayerSize, outputLayerSize, hiddenLayerSize, η, errorThreshold, iterationThreshold) {
    
    override fun buildNetwork() {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun learn() {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun calculateOutput(dataVector: DataVector) {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun calculateHiddenLayer() {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }
}