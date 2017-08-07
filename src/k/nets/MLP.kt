package k.nets

import k.neurons.HyperbolicTangentNeuron
import k.neurons.InputNeuron
import k.neurons.Neuron
import k.utils.DataVector
import k.utils.format

open class MLP(dataFileName: String,
               trainTestDivide: Int,
               inputLayerSize: Int,
               outputLayerSize: Int,
               val hiddenLayerSize: Int,
               val η: Double,
               val errorThreshold: Double,
               val iterationThreshold: Int
) : AbstractNeuralNetwork(dataFileName, trainTestDivide, inputLayerSize, outputLayerSize) {

    val hiddenLayer: ArrayList<Neuron> = ArrayList(hiddenLayerSize)

    override fun buildNetwork() {
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
    override fun learn() {
        backPropagation()
    }

    override fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateHiddenLayer()
        calculateOutputLayer()
    }

    override fun calculateHiddenLayer() {
        for (neuron: Neuron in hiddenLayer) {
            neuron.calculateState()
        }
    }

    fun backPropagation() {
        var iteration: Int = 0
        var prevError: Double = Double.MAX_VALUE
        var errorDiff: Double

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
            var curError: Double = 0.0

            for (dataVector: DataVector in trainData) {
                calculateOutput(dataVector)
                val result = getOutputValue()
                for (i in dataVector.Forecast.indices) {
                    curError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                }
            }
            curError = Math.sqrt(curError / (trainData.size))
            errorDiff = Math.abs(prevError - curError)

            System.out.println("Пред: ${prevError.format(6)} Тек: ${curError.format(6)} Раз: ${errorDiff.format(6)} Итер: $iteration")

            iteration++
            prevError = curError
        } while (errorThreshold < errorDiff && iterationThreshold > iteration)
    }
}