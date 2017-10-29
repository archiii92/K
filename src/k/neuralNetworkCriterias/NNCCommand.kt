package k.neuralNetworkCriterias

import k.nets.NeuralNetwork
import k.utils.DataVector

interface NNCCommand {
    fun calculateCriteria(data: ArrayList<DataVector>, neuralNetwork: NeuralNetwork): Double
}