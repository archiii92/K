package k.nets

import k.neurons.AbstractMLPNeuron
import k.neurons.HyperbolicTangentNeuron
import k.neurons.InputNeuron
import k.neurons.Neuron
import k.utils.DataVector
import k.utils.denormalized
import k.utils.format
import k.utils.normalized

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

    val inputLayer: ArrayList<Neuron> = ArrayList(inputLayerSize)
    val hiddenLayer: ArrayList<AbstractMLPNeuron> = ArrayList(hiddenLayerSize)
    val outputLayer: ArrayList<AbstractMLPNeuron> = ArrayList(outputLayerSize)

    final override fun prepareData() {
        k.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
    }

    override fun buildNetwork() {
        var i: Int = 0
        while (i < inputLayerSize) {
            val inputNeuron: Neuron = InputNeuron()
            inputLayer.add(inputNeuron)
            i++
        }

        i = 0
        while (i < hiddenLayerSize) {
            val hyperbolicTangentNeuron: AbstractMLPNeuron = HyperbolicTangentNeuron(inputLayer)
            hiddenLayer.add(hyperbolicTangentNeuron)
            i++
        }

        i = 0
        while (i < outputLayerSize) {
            val outputNeuron: AbstractMLPNeuron = HyperbolicTangentNeuron(hiddenLayer)
            outputLayer.add(outputNeuron)
            i++
        }
    }

    /* Обучение производится с помощью алгоритма обратного распространения ошибки */
    override fun learn() {
        backPropagation()
    }

    final override fun test() {
        var trainError: Double = 0.0
        var testError: Double = 0.0

        for (dataVector: DataVector in trainData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                trainError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println("Знач: ${dataVector.Forecast[i].format(3)} Прог: ${result[i].format(3)}")
            }
        }
        trainError = Math.sqrt(trainError / (trainData.size))

        for (dataVector: DataVector in testData) {
            calculateOutput(dataVector)
            val result = getOutputValue()
            for (i in dataVector.Forecast.indices) {
                testError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println("Знач: ${dataVector.Forecast[i].format(3)} Прог: ${result[i].format(3)}")
            }
        }
        testError = Math.sqrt(testError / (testData.size))

        System.out.println("Трен: ${trainError.format(6)} Тест: ${testError.format(6)}")
    }

    final override fun setInputValue(dataVector: DataVector) {
        for (i in inputLayer.indices) {
            val normalizedValue = normalized(dataVector.Window[i])
            inputLayer[i].value = normalizedValue
        }
    }

    override fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateHiddenLayer()
        calculateOutputLayer()
    }

    final override fun calculateHiddenLayer() {
        hiddenLayer.forEach { neuron: Neuron -> neuron.calculateState() }
    }

    final override fun calculateOutputLayer() {
        outputLayer.forEach { neuron: Neuron -> neuron.calculateState() }
    }

    final override fun getOutputValue(): ArrayList<Double> {
        return ArrayList(outputLayer.map { x -> denormalized(x.value) })
    }

    private fun backPropagation() {
        var iteration: Int = 0
        var prevError: Double = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        do {
            for (dataVector: DataVector in trainData) {

                calculateOutput(dataVector)
                val result: ArrayList<Double> = getOutputValue()
                // Обратный проход сигнала через выходной слой
                for (i in outputLayer.indices) {
                    val outputNeuron: AbstractMLPNeuron = outputLayer[i]
                    // δ = (y - d) * (df(u2) / du2)
                    outputNeuron.δ = (result[i] - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

                    // ΔW = - η * δ * v
                    //outputNeuron.ΔW = ArrayList(hiddenLayerSize)

                    for (j in outputNeuron.prevLayer.indices) {
                        outputNeuron.ΔW[j] = -η * outputNeuron.δ * outputNeuron.prevLayer[j].value
                    }
                }

                // Обратный проход сигнала через скрытый слой
                for (i in hiddenLayer.indices) {
                    val hiddenNeuron: AbstractMLPNeuron = hiddenLayer[i]

                    // δ = ∑(y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                    hiddenNeuron.δ = 0.0
                    for (j in outputLayer.indices) {
                        hiddenNeuron.δ += outputLayer[j].δ * hiddenNeuron.weights[j]
                    }
                    hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                    // ΔW = - η * δ * x
                    //hiddenNeuron.ΔW = ArrayList(inputLayerSize)

                    for (j in hiddenNeuron.prevLayer.indices) {
                        hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.prevLayer[j].value
                    }
                }

                // Уточнение весов скрытого слоя
                for (hiddenNeuron: AbstractMLPNeuron in hiddenLayer) {
                    for (i in hiddenNeuron.weights.indices) {
                        hiddenNeuron.weights[i] += hiddenNeuron.ΔW[i]
                    }
                }

                // Уточнение весов входного слоя
                for (outputNeuron: AbstractMLPNeuron in outputLayer) {
                    for (i in outputNeuron.weights.indices) {
                        outputNeuron.weights[i] += outputNeuron.ΔW[i]
                    }
                }
            }
            curError = 0.0

            for (dataVector: DataVector in trainData) {
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