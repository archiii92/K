package k.nets

import k.neurons.AbstractMLPNeuron
import k.neurons.HyperbolicTangentNeuron
import k.neurons.InputNeuron
import k.neurons.Neuron
import k.utils.DataVector
import k.utils.denormalize
import k.utils.format
import k.utils.normalize

open class MLP(
        override val dataFileName: String,
        override val trainTestDivide: Int,
        val inputLayerSize: Int,
        val hiddenLayerSize: Int,
        val outputLayerSize: Int,
        val η: Double,
        val errorThresholdBackPropagation: Double,
        val iterationThresholdBackPropagation: Int
) : NeuralNetwork {
    override val trainData: ArrayList<DataVector> = ArrayList()
    override val testData: ArrayList<DataVector> = ArrayList()

    open val inputLayer: ArrayList<Neuron> = ArrayList(inputLayerSize + 1)
    val hiddenLayer: ArrayList<Neuron> = ArrayList(hiddenLayerSize + 1)
    val outputLayer: ArrayList<Neuron> = ArrayList(outputLayerSize)

    final override fun prepareData() {
        k.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
    }

    override fun buildNetwork() {
        var i = 0
        while (i < inputLayerSize) {
            val inputNeuron = InputNeuron()
            inputLayer.add(inputNeuron)
            i++
        }
        inputLayer.add(InputNeuron(1.0))

        i = 0
        while (i < hiddenLayerSize) {
            val hyperbolicTangentNeuron = HyperbolicTangentNeuron(inputLayer)
            hiddenLayer.add(hyperbolicTangentNeuron)
            i++
        }
        hiddenLayer.add(InputNeuron(1.0))

        i = 0
        while (i < outputLayerSize) {
            val outputNeuron = HyperbolicTangentNeuron(hiddenLayer)
            outputLayer.add(outputNeuron)
            i++
        }
    }

    /* Обучение производится с помощью алгоритма обратного распространения ошибки */
    override fun learn() {
        backPropagation()
    }

    final override fun test() {
        var trainError = 0.0
        var testError = 0.0

        for (dataVector in trainData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                trainError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println("Знач: ${dataVector.Forecast[i].format(3)} Прог: ${result[i].format(3)}")
            }
        }
        trainError = Math.sqrt(trainError / trainData.size)

        for (dataVector in testData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                testError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println("Знач: ${dataVector.Forecast[i].format(3)} Прог: ${result[i].format(3)}")
            }
        }
        testError = Math.sqrt(testError / testData.size)

        System.out.println("Трен: ${trainError.format(6)} Тест: ${testError.format(6)}")
    }

    override fun setInputValue(dataVector: DataVector) {
        var i = 0
        while (i < inputLayerSize) {
            val normalizedValue = normalize(dataVector.Window[i])
            inputLayer[i].value = normalizedValue
            i++
        }
    }

    override fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateHiddenLayer()
        calculateOutputLayer()
    }

    final override fun calculateHiddenLayer() {
        hiddenLayer.forEach { neuron -> neuron.calculateState() }
    }

    final override fun calculateOutputLayer() {
        outputLayer.forEach { neuron -> neuron.calculateState() }
    }

    override fun getOutputValue(): DoubleArray {
        val result = DoubleArray(outputLayer.size)
        for (i in outputLayer.indices) {
            result[i] = denormalize(outputLayer[i].value)
        }
        return result
    }

    private fun backPropagation() {
        var iteration = 0
        var prevError = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        do {
            for (dataVector in trainData) {

                calculateOutput(dataVector)
                val result = getOutputValue()
                // Обратный проход сигнала через выходной слой
                for (i in outputLayer.indices) {
                    val outputNeuron = outputLayer[i] as AbstractMLPNeuron
                    // δ = (y - d) * (df(u2) / du2)
                    outputNeuron.δ = (result[i] - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

                    // ΔW = - η * δ * v
                    for (j in outputNeuron.prevLayer.indices) {
                        outputNeuron.ΔW[j] = -η * outputNeuron.δ * outputNeuron.prevLayer[j].value
                    }
                }

                // Обратный проход сигнала через скрытый слой
                var i = 0
                while (i < hiddenLayerSize) {
                    val hiddenNeuron = hiddenLayer[i] as AbstractMLPNeuron

                    // δ = ∑(y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                    hiddenNeuron.δ = 0.0
                    for (s in outputLayer.indices) {
                        val outputNeuron = outputLayer[s] as AbstractMLPNeuron
                        hiddenNeuron.δ += outputNeuron.δ * outputNeuron.weights[i]
                    }
                    hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                    // ΔW = - η * δ * x
                    for (j in hiddenNeuron.prevLayer.indices) {
                        hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.prevLayer[j].value
                    }
                    i++
                }

                // Уточнение весов вsходного слоя
                i = 0
                while (i < outputLayerSize) {
                    val outputNeuron = outputLayer[i] as AbstractMLPNeuron
                    for (j in outputNeuron.weights.indices) {
                        outputNeuron.weights[j] += outputNeuron.ΔW[j]
                    }
                    i++
                }

                // Уточнение весов скрытого слоя
                i = 0
                while (i < hiddenLayerSize) {
                    val hiddenNeuron = hiddenLayer[i] as AbstractMLPNeuron
                    for (j in hiddenNeuron.weights.indices) {
                        hiddenNeuron.weights[j] += hiddenNeuron.ΔW[j]
                    }
                    i++
                }
            }
            curError = 0.0

            for (dataVector in trainData) {
                calculateOutput(dataVector)
                val result = getOutputValue()
                for (i in dataVector.Forecast.indices) {
                    curError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                }
            }
            curError = Math.sqrt(curError / (trainData.size))
            errorDiff = Math.abs(prevError - curError)

            iteration++
            prevError = curError

            System.out.println("Пред: ${prevError.format(6)} Тек: ${curError.format(6)} Раз: ${errorDiff.format(6)} Итер: $iteration")
        } while (errorThresholdBackPropagation < errorDiff && iterationThresholdBackPropagation > iteration)
    }
}