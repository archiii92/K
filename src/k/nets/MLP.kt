package k.nets

import k.neurons.InputNeuron
import k.neurons.LogisticNeuron
import k.neurons.Neuron
import k.utils.DataVector

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
        val errorThreshold: Double = 5e-6, // Желаемая погрешность 5 * 10 ^ -6
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
            val hyperbolicTangentNeuron: Neuron = LogisticNeuron(inputLayer)
            hiddenLayer.add(hyperbolicTangentNeuron)
            i++
        }

        i = 0
        while (i < outputLayerSize) {
            val outputNeuron: Neuron = LogisticNeuron(hiddenLayer)
            outputLayer.add(outputNeuron)
            i++
        }
    }

    /* Обучение производится с помощью алгоритма обратного распространения ошибки */
    fun learn() {
        var iteration: Int = 0
        var error: Double = 0.0

        do {
            for (dataVector: DataVector in trainData) {

                calculateOutput(dataVector)

                // Обратный проход сигнала через выходной слой
                for (i in outputLayer.indices) {
                    val outputNeuron: Neuron = outputLayer[i]
                    // δ = (y - d) * (df(u2) / du2)
                    outputNeuron.δ = (outputNeuron.value - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

                    // ΔW = - η * δ * v
                    outputNeuron.ΔW = ArrayList(hiddenLayerSize)

                    for (j in outputNeuron.prevLayer.indices) {
                        outputNeuron.ΔW.add(-η * outputNeuron.δ * outputNeuron.prevLayer[j].value)
                    }
                }

                // Обратный проход сигнала через скрытый слой
                for (i in hiddenLayer.indices) {
                    val hiddenNeuron: Neuron = hiddenLayer[i]

                    // δ = Σ (y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                    hiddenNeuron.δ = 0.0
                    for (j in outputLayer.indices) {
                        hiddenNeuron.δ += outputLayer[j].δ * hiddenNeuron.weights[j]
                    }
                    hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                    // ΔW = - η * δ * x
                    hiddenNeuron.ΔW = ArrayList(inputLayerSize)

                    for (j in hiddenNeuron.prevLayer.indices) {
                        hiddenNeuron.ΔW.add(-η * hiddenNeuron.δ * hiddenNeuron.prevLayer[j].value)
                    }
                }

                // Уточнение весов скрытого слоя
                for (hiddenNeuron: Neuron in hiddenLayer) {
                    for (i in hiddenNeuron.weights.indices) {
                        hiddenNeuron.weights[i] += hiddenNeuron.ΔW[i]
                    }
                }

                // Уточнение весов входного слоя
                for (outputNeuron: Neuron in outputLayer) {
                    for (i in outputNeuron.weights.indices) {
                        outputNeuron.weights[i] += outputNeuron.ΔW[i]
                    }
                }
            }
            error = 0.0

            for (dataVector: DataVector in trainData) {
                calculateOutput(dataVector)
                val result = getOutputValue()
                for (i in dataVector.Forecast.indices) {
                    error += Math.pow(result[i] - dataVector.Forecast[i], 2.0) / 2
                }
            }
            error = error / (trainData.size)
            error = Math.pow(error, 0.5)
            System.out.println("Ошибка: " + error + " Итерация: " + iteration)

            iteration++
        } while (errorThreshold < error && iterationThreshold > iteration)
    }


    private fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateHiddenLayer()
        calculateOutputLayer()
        //return getOutputValue()
    }

    private fun setInputValue(dataVector: DataVector) {
        for (i in inputLayer.indices) inputLayer[i].value = dataVector.Window[i]
    }

    private fun calculateHiddenLayer() {
        for (neuron: Neuron in hiddenLayer) {
            neuron.calculateState()
        }
    }

    private fun calculateOutputLayer() {
        for (neuron: Neuron in outputLayer) {
            neuron.calculateState()
        }
    }

    private fun getOutputValue(): ArrayList<Double> {
        return ArrayList(outputLayer.map { x -> x.value })
    }
}