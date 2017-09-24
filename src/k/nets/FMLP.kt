package k.nets

import k.layers.FuzzyLayer
import k.layers.HiddenLayer
import k.layers.Layer
import k.neuronFactories.AbstractNeuronFactory
import k.neurons.AbstractMLPNeuron
import k.neurons.GaussianNeuron
import k.utils.DataVector
import k.utils.format
import k.utils.getEuclideanDistance
import java.util.*

class FMLP(
        dataFileName: String,
        trainTestDivide: Int,
        inputLayerSize: Int,
        val fuzzyLayerSize: Int,
        hiddenLayerSize: Int,
        outputLayerSize: Int,
        val errorThresholdFuzzyCMeans: Double,
        val iterationThresholdFuzzyCMeans: Int,
        η: Double,
        errorThresholdBackPropagation: Double,
        iterationThresholdBackPropagation: Int,
        val neighborsCount: Int,
        neuronFactory: AbstractNeuronFactory,
        override val showLogs: Boolean = false
) : MLP(dataFileName, trainTestDivide, inputLayerSize, hiddenLayerSize, outputLayerSize, η, errorThresholdBackPropagation, iterationThresholdBackPropagation, neuronFactory, showLogs), IFMLP {

    override val inputLayer: Layer = FuzzyLayer(fuzzyLayerSize, inputLayerSize)
    override val hiddenLayer: Layer = HiddenLayer(hiddenLayerSize, fuzzyLayerSize, neuronFactory)

    override fun clearNetwork() {
        hiddenLayer.clear()
        outputLayer.clear()
    }

    override fun initFuzzyLayer(){
        fuzzyCMeans()
        calcRadiuses(neighborsCount)
    }

    private fun fuzzyCMeans() {
        val u: Array<DoubleArray> = Array(fuzzyLayerSize) { DoubleArray(trainData.size) }
        val m = 2.0 // m — это весовой коэффициент, который принимает значения из интервала [1, ∞), на практике часто принимают m = 2

        fillUMatrix(u)
        initCenters(u, m)
    }

    // Случайная инициализация коэффициентов матрицы u, выбирая их значения из интервала [0,1] таким образом, чтобы соблюдать условие ∑u = 1
    private fun fillUMatrix(u: Array<DoubleArray>) {
        val r = Random()
        val vector = DoubleArray(fuzzyLayerSize)

        for (t in trainData.indices) {
            var i = 0
            var sum = 0.0
            while (i < fuzzyLayerSize) {
                val value: Double = r.nextDouble()
                vector[i] = value
                sum += value
                i++
            }

            for (j in vector.indices) {
                u[j][t] = vector[j] / sum // Нормируем значения коэффициентов, таким образом, чтобы они были от 0 до 1 и в сумме давали 1
            }
        }
    }

    private fun initCenters(u: Array<DoubleArray>, m: Double) {
        var iteration = 0
        var prevError = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        while (true) {
            // Определить M центров c - ci = ∑(uit) ^ m * xt / ∑(uit) ^ m
            for ((i, fuzzyNeuron) in inputLayer.neurons.withIndex()) {
                if (fuzzyNeuron is GaussianNeuron) {
                    fuzzyNeuron.center = DoubleArray(fuzzyNeuron.center.size)
                    var denominator = 0.0
                    for (t in trainData.indices) {
                        val uit = Math.pow(u[i][t], m)
                        for (j in fuzzyNeuron.center.indices) {
                            fuzzyNeuron.center[j] += uit * trainData[t].Window[j]
                        }
                        denominator += uit
                    }

                    for (j in fuzzyNeuron.center.indices) {
                        fuzzyNeuron.center[j] /= denominator
                    }
                }
            }

            // Рассчитать значение функции погрешности E = ∑∑(uit) ^ m * dit ^ 2
            curError = 0.0
            for ((i, fuzzyNeuron) in inputLayer.neurons.withIndex()) {
                if (fuzzyNeuron is GaussianNeuron) {
                    for (t in trainData.indices) {
                        curError += Math.pow(u[i][t], m) * Math.pow(getEuclideanDistance(fuzzyNeuron.center, trainData[t].Window), 2.0)
                    }
                }
            }

            errorDiff = Math.abs(prevError - curError)
            iteration++
            if (showLogs) System.out.println("Пред: ${prevError.format(7)} Тек: ${curError.format(7)} Раз: ${errorDiff.format(7)} Итер: $iteration")
            prevError = curError

            if (errorThresholdFuzzyCMeans > errorDiff && iterationThresholdFuzzyCMeans <= iteration) {
                break
            }

            // Рассчитать новые значения u по формуле 1 / ∑ (dit ^ 2 / dkt ^ 2) ^ (1 / (m - 1))
            for ((i, fuzzyNeuron) in inputLayer.neurons.withIndex()) {
                if (fuzzyNeuron is GaussianNeuron) {
                    for (t in trainData.indices) {
                        val dit: Double = Math.pow(getEuclideanDistance(fuzzyNeuron.center, trainData[t].Window), 2.0)
                        var denominator = 0.0

                        for (fuzzyNeuronK in inputLayer.neurons) {
                            if (fuzzyNeuronK is GaussianNeuron) {
                                val dkt: Double = Math.pow(getEuclideanDistance(fuzzyNeuronK.center, trainData[t].Window), 2.0)
                                denominator += Math.pow(dit / dkt, 1 / (m - 1))
                            }
                        }
                        u[i][t] = 1 / denominator
                    }
                }
            }
        }
    }

    private fun calcRadiuses(neighborsCount: Int) {
        for (fuzzyNeuron in inputLayer.neurons) {
            if (fuzzyNeuron is GaussianNeuron) {
                val norms = DoubleArray(fuzzyLayerSize - 1)

                var q = 0
                for (fuzzyNeuronJ in inputLayer.neurons) {
                    if (fuzzyNeuronJ is GaussianNeuron) {
                        if (fuzzyNeuron != fuzzyNeuronJ) {
                            norms[q] = getEuclideanDistance(fuzzyNeuron.center, fuzzyNeuronJ.center)
                            q++
                        }
                    }
                }

                norms.sort()
                var radius = 0.0
                var k = 0
                while (k < neighborsCount && k < norms.size) {
                    radius += Math.pow(norms[k], 2.0)
                    k++
                }

                radius = Math.sqrt(radius / k)
                fuzzyNeuron.radius = radius
            }
        }
    }

    override fun backPropagation() {
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

                for ((i, fuzzyNeuron) in inputLayer.neurons.withIndex()) {
                    if (fuzzyNeuron is GaussianNeuron) {
                        var sum = 0.0
                        var k = 0
                        while (k < hiddenLayerSize) {
                            val hiddenNeuron = hiddenLayer.neurons[k] as AbstractMLPNeuron
                            sum += hiddenNeuron.δ * hiddenNeuron.weights[i] * fuzzyNeuron.outputValue
                            k++
                        }
                        //constant /= hiddenLayer.size

                        fuzzyNeuron.dEdr = 0.0
                        for (j in fuzzyNeuron.center.indices) {
                            fuzzyNeuron.dEdc[j] = sum * (fuzzyNeuron.inputVector[j] - fuzzyNeuron.center[j]) / Math.pow(fuzzyNeuron.radius, 2.0)
                            fuzzyNeuron.dEdr += fuzzyNeuron.dEdc[j] / fuzzyNeuron.radius
                        }

                        fuzzyNeuron.dEdr /= fuzzyNeuron.center.size
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

                // Уточнение центра и радиуса нечеткого слоя
                for ((i, fuzzyNeuron) in inputLayer.neurons.withIndex()) {
                    if (fuzzyNeuron is GaussianNeuron) {
                        for (k in fuzzyNeuron.center.indices) {
                            fuzzyNeuron.center[k] += -η * fuzzyNeuron.dEdc[k]
                        }
                        fuzzyNeuron.radius += -η * fuzzyNeuron.dEdr
                    }
                }
            }
            curError = 0.0

            for (dataVector: DataVector in trainData) {
                val result = calculate(dataVector)
                for (i in dataVector.Forecast.indices) {
                    curError += Math.pow(result[i] - dataVector.Forecast[i], 2.0)
                }
            }
            curError = Math.sqrt(curError / (trainData.size))
            errorDiff = Math.abs(prevError - curError)

            iteration++
            prevError = curError

            if (showLogs) println("Пред: ${prevError.format(8)} Тек: ${curError.format(8)} Раз: ${errorDiff.format(8)} Итер: $iteration")
        } while (errorThresholdBackPropagation < errorDiff && iterationThresholdBackPropagation > iteration)
    }
}