package k.nets

import k.layers.FuzzyLayer
import k.layers.HiddenLayer
import k.layers.Layer
import k.neuronFactories.AbstractNeuronFactory
import k.neurons.AbstractMLPNeuron
import k.neurons.GaussianNeuron
import k.neurons.Neuron
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
        inputLayer.clear()
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

            if (errorThresholdFuzzyCMeans > errorDiff || iterationThresholdFuzzyCMeans <= iteration) {
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
                val normsAndNeurons = TreeMap<Double, Neuron>()

                for (fuzzyNeuronJ in inputLayer.neurons) {
                    if (fuzzyNeuronJ is GaussianNeuron) {
                        if (fuzzyNeuron != fuzzyNeuronJ) {
                            val norma = getEuclideanDistance(fuzzyNeuron.center, fuzzyNeuronJ.center)
                            normsAndNeurons.set(norma, fuzzyNeuronJ)
                        }
                    }
                }

                val radius = DoubleArray(inputLayerSize)

                val neurons = normsAndNeurons.values
                val itr = neurons.iterator()
                var k = 0
                while (k < neighborsCount && itr.hasNext()) {
                    val neighborsNeuron = itr.next()
                    if (neighborsNeuron is GaussianNeuron)
                        for (i in neighborsNeuron.center.indices) {
                            radius[i] += Math.pow(fuzzyNeuron.center[i] - neighborsNeuron.center[i], 2.0)
                        }
                    k++
                }

                for (i in fuzzyNeuron.radius.indices) {
                    fuzzyNeuron.radius[i] = Math.sqrt(radius[i] / k)
                }
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
                        // δ2 = (y - d) * (df(u2) / du2)
                        outputNeuron.δ = (result[i] - dataVector.Forecast[i]) * outputNeuron.activationFunctionDerivative(outputNeuron.sum)

                        // ΔW = - η * δ2 * v
                        for (j in outputNeuron.inputVector.indices) {
                            outputNeuron.ΔW[j] = -η * outputNeuron.δ * outputNeuron.inputVector[j]
                        }
                    }
                }

                // Обратный проход сигнала через скрытый слой
                for ((i, hiddenNeuron) in hiddenLayer.neurons.withIndex()) {
                    if (hiddenNeuron is AbstractMLPNeuron){
                        // δ1 = ∑(w2 * δ2) * (df(u1) / du1)
                        hiddenNeuron.δ = 0.0
                        for (outputNeuron in outputLayer.neurons) {
                            if (outputNeuron is AbstractMLPNeuron) hiddenNeuron.δ += outputNeuron.δ * outputNeuron.weights[i]
                        }
                        hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                        // ΔW = - η * δ1 * x
                        for (j in hiddenNeuron.inputVector.indices) {
                            hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.inputVector[j]
                        }
                    }
                }

                for ((i, fuzzyNeuron) in inputLayer.neurons.withIndex()) {
                    if (fuzzyNeuron is GaussianNeuron) {
                        // δ0 = ∑(δ1 * w1) * exp(-u(t) / 2) * (x - c) / (r ^ 2)
                        fuzzyNeuron.δ = 0.0
                        for (hiddenNeuron in hiddenLayer.neurons) {
                            if (hiddenNeuron is AbstractMLPNeuron) fuzzyNeuron.δ += hiddenNeuron.δ * hiddenNeuron.weights[i]
                        }

                        fuzzyNeuron.δ *= fuzzyNeuron.outputValue

                        // Δc = - η * δ0
                        // Δr = Δc / r
                        for (j in fuzzyNeuron.center.indices) {
                            fuzzyNeuron.Δc[j] = -η * fuzzyNeuron.δ * (fuzzyNeuron.inputVector[j] - fuzzyNeuron.center[j]) / Math.pow(fuzzyNeuron.radius[j], 2.0)
                            fuzzyNeuron.Δr[j] = fuzzyNeuron.Δc[j] / fuzzyNeuron.radius[j]
                        }
                    }
                }

                // Уточнение весов вsходного слоя
                for (outputNeuron in outputLayer.neurons) {
                    if (outputNeuron is AbstractMLPNeuron) {
                        for (i in outputNeuron.weights.indices) {
                            outputNeuron.weights[i] += outputNeuron.ΔW[i]
                        }
                    }
                }

                // Уточнение весов скрытого слоя
                for (hiddenNeuron in hiddenLayer.neurons) {
                    if (hiddenNeuron is AbstractMLPNeuron) {
                        for (i in hiddenNeuron.weights.indices) {
                            hiddenNeuron.weights[i] += hiddenNeuron.ΔW[i]
                        }
                    }
                }

                // Уточнение центра и радиуса нечеткого слоя
                for (fuzzyNeuron in inputLayer.neurons) {
                    if (fuzzyNeuron is GaussianNeuron) {
                        for (i in fuzzyNeuron.center.indices) {
                            fuzzyNeuron.center[i] += fuzzyNeuron.Δc[i]
                            fuzzyNeuron.radius[i] += fuzzyNeuron.Δr[i]
                        }
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