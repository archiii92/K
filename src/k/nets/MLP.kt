package k.nets

import k.neurons.InputNeuron
import k.neurons.LogisticNeuron
import k.neurons.Neuron
import k.neurons.OutputNeuron
import k.utils.DataVector
import java.util.*

class MLP constructor(
        /* Настройка сети */
        val inputLayerSize: Int = 3, // Число входных нейронов = размер скользящего окна
        val hiddenLayerSize: Int = 12, // Число нейронов скрытого слоя
        val outputLayerSize: Int = 1, // Число выходных нейронов = размер прогноза

        /* Настройка данных */
        val dataFileName: String = "gold", // Название файла с данными
        val trainTestDivide: Int = 90, // Процент деления обучающего и тестового набора

        /* Настройка обучения */
        val η: Double = 0.1, // Коэффициент обучения
        val errorThreshold: Double = 5e-6, // Желаемая погрешность 5*10^-6
        val iterationThreshold: Int = 30000
        ) {
    val trainData: ArrayList<DataVector> = ArrayList()
    val testData: ArrayList<DataVector> = ArrayList()

    val inputLayer: ArrayList<Neuron> = ArrayList(inputLayerSize)
    val hiddenLayer: ArrayList<Neuron> = ArrayList(hiddenLayerSize)
    val outputLayer: ArrayList<Neuron> = ArrayList(outputLayerSize)

    fun prepareData() {
        k.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
    }

    fun buildNetwork() {
        var i: Int = 0
        while (i < inputLayerSize) {
            val inputNeuron: Neuron = InputNeuron()
            inputLayer.add(inputNeuron)
            i++
        }

        i = 0
        while (i < hiddenLayerSize) {
            val hyperbolicTangentNeuron: Neuron = LogisticNeuron(inputLayerSize)
            hiddenLayer.add(hyperbolicTangentNeuron)
            i++
        }

        i = 0
        while (i < outputLayerSize) {
            val outputNeuron: Neuron = OutputNeuron(hiddenLayerSize)
            outputLayer.add(outputNeuron)
            i++
        }
    }

    /* Обучение производится с помощью алгоритма обратного распространения ошибки */
    fun learn() {
        var train: Boolean = true
        var iteration: Int = 0
        var error: Double = 0.0

        while (train) {
            for (dataVector: DataVector in trainData) {

                calculateOutput(dataVector)


            }
            iteration++
        }
    }


    private fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateHiddenLayer()
    }

    private fun setInputValue(dataVector: DataVector) {
        for (i in inputLayer.indices) inputLayer[i].value = dataVector.Forecast[i]
    }

    private fun calculateHiddenLayer() {
        for (neuron: Neuron in hiddenLayer) {
            neuron.calculateState()
        }
    }
}