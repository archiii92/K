package k.nets

import k.layers.HiddenLayer
import k.layers.InputLayer
import k.layers.Layer
import k.layers.OutputLayer
import k.neuronFactories.AbstractNeuronFactory
import k.neurons.AbstractMLPNeuron
import k.utils.DataVector
import k.utils.format

open class MLP(
        override val dataFileName: String,
        override val trainTestDivide: Int,
        val inputLayerSize: Int,
        val hiddenLayerSize: Int,
        val outputLayerSize: Int,
        val η: Double,
        val errorThresholdBackPropagation: Double,
        val iterationThresholdBackPropagation: Int,
        val neuronFactory: AbstractNeuronFactory
) : NeuralNetwork {
    override val trainData: ArrayList<DataVector> = ArrayList()
    override val testData: ArrayList<DataVector> = ArrayList()

    open val inputLayer: Layer = InputLayer(inputLayerSize)
    open val hiddenLayer: Layer = HiddenLayer(hiddenLayerSize, inputLayerSize, neuronFactory)
    val outputLayer: Layer = OutputLayer(outputLayerSize, hiddenLayerSize, neuronFactory)

    final override fun prepareData() {
        k.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
    }

    override fun buildNetwork() {
        inputLayer.build()
        hiddenLayer.build()
        outputLayer.build()
    }

    override fun learn() {
        backPropagation()
    }

    final override fun test() {
        var trainError = 0.0
        var testError = 0.0

        for (dataVector in trainData) {
            val result = calculate(dataVector)
            for (i in dataVector.Forecast.indices) {
                trainError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println("Знач: ${dataVector.Forecast[i].format(3)} Прог: ${result[i].format(3)}")
            }
        }
        trainError = Math.sqrt(trainError / trainData.size)

        for (dataVector in testData) {
            val result = calculate(dataVector)
            for (i in dataVector.Forecast.indices) {
                testError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                System.out.println("Знач: ${dataVector.Forecast[i].format(3)} Прог: ${result[i].format(3)}")
            }
        }
        testError = Math.sqrt(testError / testData.size)

        System.out.println("Трен: ${trainError.format(6)} Тест: ${testError.format(6)}")
    }

    override fun calculate(dataVector: DataVector): DoubleArray {
        inputLayer.inputVector = dataVector.Window
        inputLayer.calculate()
        hiddenLayer.inputVector = inputLayer.outputVector
        hiddenLayer.calculate()
        outputLayer.inputVector = hiddenLayer.outputVector
        outputLayer.calculate()
        return outputLayer.outputVector
    }

    private fun backPropagation() {
        var iteration = 0
        var prevError = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        do {
            for (dataVector in trainData) {

                val result = calculate(dataVector)
                // Обратный проход сигнала через выходной слой
                var i = 0
                while (i < outputLayerSize) {
                    val outputNeuron = outputLayer.neurons[i] as AbstractMLPNeuron
                    // δ = (y - d) * (df(u2) / du2)
                    outputNeuron.δ = (result[i] - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

                    // ΔW = - η * δ * v
                    for (j in outputNeuron.inputVector.indices) {
                        outputNeuron.ΔW[j] = -η * outputNeuron.δ * outputNeuron.inputVector[j]
                    }
                    i++
                }

                // Обратный проход сигнала через скрытый слой
                i = 0
                while (i < hiddenLayerSize) {
                    val hiddenNeuron = hiddenLayer.neurons[i] as AbstractMLPNeuron

                    // δ = ∑(y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                    hiddenNeuron.δ = 0.0
                    for (s in outputLayer.neurons.indices) {
                        val outputNeuron = outputLayer.neurons[s] as AbstractMLPNeuron
                        hiddenNeuron.δ += outputNeuron.δ * outputNeuron.weights[i]
                    }
                    hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                    // ΔW = - η * δ * x
                    for (j in hiddenNeuron.inputVector.indices) {
                        hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.inputVector[j]
                    }
                    i++
                }

                // Уточнение весов вsходного слоя
                i = 0
                while (i < outputLayerSize) {
                    val outputNeuron = outputLayer.neurons[i] as AbstractMLPNeuron
                    for (j in outputNeuron.weights.indices) {
                        outputNeuron.weights[j] += outputNeuron.ΔW[j]
                    }
                    i++
                }

                // Уточнение весов скрытого слоя
                i = 0
                while (i < hiddenLayerSize) {
                    val hiddenNeuron = hiddenLayer.neurons[i] as AbstractMLPNeuron
                    for (j in hiddenNeuron.weights.indices) {
                        hiddenNeuron.weights[j] += hiddenNeuron.ΔW[j]
                    }
                    i++
                }
            }
            curError = 0.0

            for (dataVector in trainData) {
                val result = calculate(dataVector)
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