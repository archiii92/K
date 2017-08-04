package k.nets

import k.neurons.HyperbolicTangentNeuron
import k.neurons.InputNeuron
import k.neurons.Neuron
import k.utils.DataVector
import k.utils.denormalized
import k.utils.normalized

class MLP(
        /* Настройка сети */
        val inputLayerSize: Int = 3, // Число входных нейронов = размер скользящего окна
        val hiddenLayerSize: Int = 6, // Число нейронов скрытого слоя
        val outputLayerSize: Int = 1, // Число выходных нейронов = размер прогноза

        /* Настройка данных */
        val dataFileName: String = "gold.txt", // Название файла с данными // gold.txt temperature.csv
        val trainTestDivide: Int = 80, // Процент деления обучающего и тестового набора

        /* Настройка обучения */
        val η: Double = 0.01, // Коэффициент обучения
        val errorThreshold: Double = 5e-6, // Желаемая погрешность 5 * 10 ^ -6
        val iterationThreshold: Int = 10000
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
            val hyperbolicTangentNeuron: Neuron = HyperbolicTangentNeuron(inputLayer)
            hiddenLayer.add(hyperbolicTangentNeuron)
            i++
        }

        i = 0
        while (i < outputLayerSize) {
            val outputNeuron: Neuron = HyperbolicTangentNeuron(hiddenLayer)
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
                val result: ArrayList<Double> = getOutputValue()
                // Обратный проход сигнала через выходной слой
                for (i in outputLayer.indices) {
                    val outputNeuron: Neuron = outputLayer[i]
                    // δ = (y - d) * (df(u2) / du2)
                    outputNeuron.δ = (result[i] - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

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
                    error += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                }
            }
            error = Math.sqrt(error / (trainData.size))
            System.out.println("Ошибка: " + error + " Итерация: " + iteration)

            iteration++
        } while (errorThreshold < error && iterationThreshold > iteration)
    }

    fun test() {
        var trainError: Double = 0.0
        var testError: Double = 0.0

        for (dataVector: DataVector in trainData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                trainError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println(String.format("%1.1f | %1.1f", dataVector.Forecast[i], result[i]))
            }
        }
        trainError = Math.sqrt(trainError / (trainData.size))

        for (dataVector: DataVector in testData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                testError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println(String.format("%1.1f | %1.1f", dataVector.Forecast[i], result[i]))
            }
        }
        testError = Math.sqrt(testError / (testData.size))

        System.out.println(String.format("Train %1.3f | Test %1.3f", trainError, testError))
    }

    private fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateHiddenLayer()
        calculateOutputLayer()
    }

    private fun setInputValue(dataVector: DataVector) {
        for (i in inputLayer.indices) {
            val normalizedValue = normalized(dataVector.Window[i])
            inputLayer[i].value = normalizedValue
        }
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
        return ArrayList(outputLayer.map { x -> denormalized(x.value) })
    }
}