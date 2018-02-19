package kotlinFMLP.nets

import kotlinFMLP.layers.HiddenLayer
import kotlinFMLP.layers.InputLayer
import kotlinFMLP.layers.Layer
import kotlinFMLP.layers.OutputLayer
import kotlinFMLP.neuralNetworkCriterias.NNCCommand
import kotlinFMLP.neuronFactories.AbstractNeuronFactory
import kotlinFMLP.neuronWeightsOptimizers.NWOCommand
import kotlinFMLP.neurons.AbstractMLPNeuron
import kotlinFMLP.utils.DataVector
import kotlinFMLP.utils.format

open class MLP(
        override val dataFileName: String,
        override val trainTestDivide: Int,
        val inputLayerSize: Int,
        val hiddenLayerSize: Int,
        val outputLayerSize: Int,
        val η: Double,
        val errorThresholdBackPropagation: Double,
        val iterationThresholdBackPropagation: Int,
        neuronFactory: AbstractNeuronFactory,
        val neuralNetworkCriteria: NNCCommand,
        override val showLogs: Boolean = false
) : NeuralNetwork {
    override val trainData: ArrayList<DataVector> = ArrayList()
    override val testData: ArrayList<DataVector> = ArrayList()

    open val inputLayer: Layer = InputLayer(inputLayerSize)
    open val hiddenLayer: Layer = HiddenLayer(hiddenLayerSize, inputLayerSize, neuronFactory)
    val outputLayer: Layer = OutputLayer(outputLayerSize, hiddenLayerSize, neuronFactory)

    final override fun prepareData() {
        kotlinFMLP.utils.readData(trainData, testData, trainTestDivide, dataFileName, inputLayerSize, outputLayerSize)
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
        val trainError = calculateError(trainData)
        val testError = calculateError(testData)

        println("Трен СКО: ${trainError.format(6)} Тест СКО: ${testError.format(6)}")
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

    override fun optimizeMLPNeuronWeigths(neuronWeightsOptimizer: NWOCommand) {
        val beforeError = calculateError(trainData)
        for (neuron in hiddenLayer.neurons) {
            if (neuron is AbstractMLPNeuron) {
                neuronWeightsOptimizer.optimizeWeights(neuron, this)
            }
        }
        for (neuron in outputLayer.neurons) {
            if (neuron is AbstractMLPNeuron) {
                neuronWeightsOptimizer.optimizeWeights(neuron, this)
            }
        }
        val afterError = calculateError(trainData)
        if (showLogs) println("Итоговое улучшение ошибки: ${(beforeError - afterError).format(6)}")
    }

    override fun calculateError(data: ArrayList<DataVector>): Double {
        return neuralNetworkCriteria.calculateCriteria(data, this)
    }

    override fun clearNetwork() {
        hiddenLayer.clear()
        outputLayer.clear()
    }

    override fun shuffleData(){
        kotlinFMLP.utils.shuffleData(this.trainData, this.testData)
    }

    open fun backPropagation() {
        var iteration = 0
        var prevError = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        do {
            for (dataVector in trainData) {
                val result = calculate(dataVector)

                // Обратный проход сигнала через выходной слой
                for ((i, outputNeuron) in outputLayer.neurons.withIndex()) {
                    if (outputNeuron is AbstractMLPNeuron){
                        // δ = (y - d) * (df(u2) / du2)
                        outputNeuron.δ = (result[i] - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

                        // ΔW = - η * δ * v
                        for (j in outputNeuron.inputVector.indices) {
                            outputNeuron.ΔW[j] = -η * outputNeuron.δ * outputNeuron.inputVector[j]
                        }
                    }
                }

                // Обратный проход сигнала через скрытый слой
                for ((i, hiddenNeuron) in hiddenLayer.neurons.withIndex()) {
                    if (hiddenNeuron is AbstractMLPNeuron){
                        // δ = ∑(y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                        hiddenNeuron.δ = 0.0
                        for (outputNeuron in outputLayer.neurons) {
                            if (outputNeuron is AbstractMLPNeuron) hiddenNeuron.δ += outputNeuron.δ * outputNeuron.weights[i]
                        }
                        hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                        // ΔW = - η * δ * x
                        for (j in hiddenNeuron.inputVector.indices) {
                            hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.inputVector[j]
                        }
                    }
                }

                // Уточнение весов вsходного слоя
                for (outputNeuron in outputLayer.neurons) {
                    if (outputNeuron is AbstractMLPNeuron) {
                        for (j in outputNeuron.weights.indices) {
                            outputNeuron.weights[j] += outputNeuron.ΔW[j]
                        }
                    }
                }

                // Уточнение весов скрытого слоя
                for (hiddenNeuron in hiddenLayer.neurons) {
                    if (hiddenNeuron is AbstractMLPNeuron) {
                        for (j in hiddenNeuron.weights.indices) {
                            hiddenNeuron.weights[j] += hiddenNeuron.ΔW[j]
                        }
                    }
                }
            }

            // При обучении ставится задача минимизации целевой функции, формируемой, как правило, в виде квадратичной
            // суммы разностей между фактическими и ожидаемыми значениями выходных сигналов E(w) = 1 / 2 * ∑∑ (( y - d ) ^ 2 )
            curError = 0.0
            for (dataVector in trainData) {
                val result = calculate(dataVector)
                for (i in dataVector.Forecast.indices) {
                    curError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                }
            }

            curError /= 2

            errorDiff = Math.abs(prevError - curError)

            iteration++
            prevError = curError

            if (showLogs) println("Пред: ${prevError.format(6)} Тек: ${curError.format(6)} Раз: ${errorDiff.format(6)} Итер: $iteration")
        } while (errorThresholdBackPropagation < errorDiff && iterationThresholdBackPropagation > iteration)
    }
}