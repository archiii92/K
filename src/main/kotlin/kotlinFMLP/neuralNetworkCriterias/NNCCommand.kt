package kotlinFMLP.neuralNetworkCriterias

import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.utils.DataVector

interface NNCCommand {
    fun calculateCriteria(data: ArrayList<DataVector>, neuralNetwork: NeuralNetwork): Double
}