package k.nets

import k.neurons.*
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
        val neighborsCount: Int
) : MLP(dataFileName, trainTestDivide, inputLayerSize, hiddenLayerSize, outputLayerSize, η, errorThresholdBackPropagation, iterationThresholdBackPropagation) {

    val fuzzyLayer: ArrayList<GaussianNeuron> = ArrayList(fuzzyLayerSize)

    override fun buildNetwork() {
        var i: Int = 0
        while (i < inputLayerSize) {
            val inputNeuron: Neuron = InputNeuron()
            inputLayer.add(inputNeuron)
            i++
        }

        i = 0
        while (i < fuzzyLayerSize) {
            val gaussianNeuron: GaussianNeuron = GaussianNeuron(inputLayer)
            fuzzyLayer.add(gaussianNeuron)
            i++
        }

        i = 0
        while (i < hiddenLayerSize) {
            val hyperbolicTangentNeuron: AbstractMLPNeuron = HyperbolicTangentNeuron(fuzzyLayer)
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

    override fun learn() {
        fuzzyCMeans()
        calcRadiuses(neighborsCount)
        backPropagation()
    }

    override fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateFuzzyLayer()
        calculateHiddenLayer()
        calculateOutputLayer()
    }

    override fun setInputValue(dataVector: DataVector) {
        for (i in inputLayer.indices) {
            inputLayer[i].value = dataVector.Window[i]
        }
    }

    private fun calculateFuzzyLayer() {
        fuzzyLayer.forEach { neuron: Neuron -> neuron.calculateState() }
    }

    private fun fuzzyCMeans() {

        val u: Array<DoubleArray> = Array(fuzzyLayerSize) { DoubleArray(trainData.size) }
        val m: Double = 2.0 // m — это весовой коэффициент, который принимает значения из интервала [1, ∞), на практике часто принимают m = 2

        fillUMatrix(u)
        initCenters(u, m)
    }

    // Случайная инициализация коэффициентов матрицы u, выбирая их значения из интервала [0,1] таким образом, чтобы соблюдать условие ∑u = 1
    private fun fillUMatrix(u: Array<DoubleArray>) {
        val r: Random = Random()

        var sum: Double = 0.0
        val vector: DoubleArray = DoubleArray(fuzzyLayerSize)
        for (t in trainData.indices) {

            var i: Int = 0
            while (i < fuzzyLayerSize) {
                val value: Double = r.nextDouble()
                vector[i] = value
                sum += value

                i++
            }

            for (j in vector.indices) {
                u[j][t] = vector[j] / sum // Нормируем значения коэффициентов, таким образом, чтобы они были от 0 до 1 и в сумме давали 1
            }

            sum = 0.0
        }
    }

    private fun initCenters(u: Array<DoubleArray>, m: Double) {
        var iteration: Int = 0
        var prevError: Double = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        while (true) {
            // Определить M центров c - ci = ∑(uit) ^ m * xt / ∑(uit) ^ m
            var i: Int = 0
            while (i < fuzzyLayerSize) {
                fuzzyLayer[i].center = DoubleArray(fuzzyLayer[i].center.size)
                i++
            }

            i = 0
            while (i < fuzzyLayerSize) {

                val center: DoubleArray = fuzzyLayer[i].center

                var denominator: Double = 0.0

                for (t in trainData.indices) {
                    val uit = Math.pow(u[i][t], m)
                    for (j in trainData[t].Window.indices) {
                        center[j] += uit * trainData[t].Window[j]
                    }
                    denominator += uit
                }

                for (j in center.indices) {
                    center[j] /= denominator
                }
                i++
            }

            // Рассчитать значение функции погрешности E = ∑∑(uit) ^ m * dit ^ 2
            i = 0
            curError = 0.0
            while (i < fuzzyLayerSize) {
                for (t in trainData.indices) {
                    curError += Math.pow(u[i][t], m) * Math.pow(getEuclideanDistance(fuzzyLayer[i].center, trainData[t].Window), 2.0)
                }
                i++
            }

            errorDiff = Math.abs(prevError - curError)

            iteration++

            System.out.println("Пред: ${prevError.format(7)} Тек: ${curError.format(7)} Раз: ${errorDiff.format(7)} Итер: $iteration")

            prevError = curError

            if (errorThresholdFuzzyCMeans > errorDiff && iterationThresholdFuzzyCMeans <= iteration) {
                break
            }

            // Рассчитать новые значения u по формуле 1 / ∑ (dit ^ 2 / dkt ^ 2) ^ (1 / (m - 1))
            i = 0
            while (i < fuzzyLayerSize) {
                for (t in trainData.indices) {

                    val dit: Double = Math.pow(getEuclideanDistance(fuzzyLayer[i].center, trainData[t].Window), 2.0)

                    var denominator: Double = 0.0
                    var k: Int = 0
                    while (k < fuzzyLayerSize) {

                        val dkt: Double = Math.pow(getEuclideanDistance(fuzzyLayer[k].center, trainData[t].Window), 2.0)
                        denominator += Math.pow(dit / dkt, 1 / (m - 1))

                        k++
                    }

                    u[i][t] = 1 / denominator
                }
                i++
            }
        }
    }

    private fun calcRadiuses(neighborsCount: Int) {
        var i: Int = 0
        while (i < fuzzyLayerSize) {

            val norms: DoubleArray = DoubleArray(fuzzyLayerSize - 1)

            var j: Int = 0
            var q = 0
            while (j < fuzzyLayerSize) {

                if (i != j) {
                    norms[q] = getEuclideanDistance(fuzzyLayer[i].center, fuzzyLayer[j].center)
                    q++
                }
                j++
            }

            norms.sort()

            var radius: Double = 0.0
            var k = 0
            while (k < neighborsCount && k < norms.size) {
                radius += Math.pow(norms[k], 2.0)
                k++
            }

            radius = Math.sqrt(radius / k)

            fuzzyLayer[i].radius = radius

            i++
        }
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
                    for (j in outputNeuron.prevLayer.indices) {
                        outputNeuron.ΔW[j] = -η * outputNeuron.δ * outputNeuron.prevLayer[j].value
                    }
                }

                // Обратный проход сигнала через скрытый слой
                for (i in hiddenLayer.indices) {
                    val hiddenNeuron: AbstractMLPNeuron = hiddenLayer[i]

                    // δ = ∑(y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                    hiddenNeuron.δ = 0.0
                    for (s in outputLayer.indices) {
                        hiddenNeuron.δ += outputLayer[s].δ * outputLayer[s].weights[i]
                    }
                    hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                    // ΔW = - η * δ * x
                    for (j in hiddenNeuron.prevLayer.indices) {
                        hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.prevLayer[j].value
                    }
                }

                for (i in fuzzyLayer.indices) {
                    val fuzzyNeuron = fuzzyLayer[i]

                    fuzzyNeuron.δ = 0.0
                    for (s in hiddenLayer.indices) {
                        fuzzyNeuron.δ += hiddenLayer[s].δ
                    }
                    fuzzyNeuron.δ *= fuzzyNeuron.value

                    for (j in fuzzyNeuron.prevLayer.indices) {
                        fuzzyNeuron.ΔW[j] = -η * fuzzyNeuron.δ
                    }

                    var constant = 0.0
                    for (s in hiddenLayer.indices) {
                        constant = hiddenLayer[s].δ * hiddenLayer[s].weights[i] * fuzzyNeuron.value
                    }

                    for (j in fuzzyNeuron.center.indices) {
                        val c = constant * (fuzzyNeuron.prevLayer[j].value - fuzzyNeuron.center[j]) / Math.pow(fuzzyNeuron.radius, 2.0)
                        val r = c / fuzzyNeuron.radius

                        fuzzyNeuron.center[j] += -η * c
                        fuzzyNeuron.radius += -η * r
                    }


                    for (i in fuzzyNeuron.weights.indices) {
                        fuzzyNeuron.weights[i] += fuzzyNeuron.ΔW[i]
                    }
                }

                // Уточнение весов скрытого слоя
                for (hiddenNeuron: AbstractMLPNeuron in hiddenLayer) {
                    for (i in hiddenNeuron.weights.indices) {
                        hiddenNeuron.weights[i] += hiddenNeuron.ΔW[i]
                    }
                }

                // Уточнение весов выходного слоя
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

            System.out.println("Пред: ${prevError.format(8)} Тек: ${curError.format(8)} Раз: ${errorDiff.format(8)} Итер: $iteration")
        } while (errorThresholdBackPropagation < errorDiff && iterationThresholdBackPropagation > iteration)
    }
}