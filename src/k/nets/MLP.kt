package k.nets

import k.neurons.Neuron
import k.utils.DataVector
import java.util.ArrayList

class MLP constructor(
        val inputLayerSize: Int = 3, // Число входных нейронов = размер скользящего окна
        val hiddenLayerSize: Int = 12, // Число нейронов скрытого слоя
        val outputLayerSize: Int = 1, // Число выходных нейронов = размер прогноза
        val dataFileName: String = "gold", // Название файла с данными
        val trainTestDivide: Int = 90 // Процент деления обучающего и тестового набора
        ) {
    val trainData: ArrayList<DataVector> = ArrayList()
    val testData: ArrayList<DataVector> = ArrayList()

    val inputLayer: ArrayList<Neuron> = ArrayList(inputLayerSize)
    val hiddenLayer: ArrayList<Neuron> = ArrayList(hiddenLayerSize)
    val outputLayer: ArrayList<Neuron> = ArrayList(outputLayerSize)

    fun prepareData() {
        k.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
    }

    fun learn() {
        var i = 0;
        while (i < inputLayerSize) {
            //inputLayer.add()
            i++
        }
    }
}