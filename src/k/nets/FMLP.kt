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

    override val inputLayer: ArrayList<Neuron> = ArrayList(inputLayerSize)
    val fuzzyLayer: ArrayList<Neuron> = ArrayList(fuzzyLayerSize - 1)

    override fun buildNetwork() {
        var i = 0
        while (i < inputLayerSize) {
            val inputNeuron = InputNeuron()
            inputLayer.add(inputNeuron)
            i++
        }

        i = 0
        while (i < fuzzyLayerSize) {
            val gaussianNeuron = GaussianNeuron(inputLayer)
            fuzzyLayer.add(gaussianNeuron)
            i++
        }
        fuzzyLayer.add(InputNeuron(1.0))

        i = 0
        while (i < hiddenLayerSize) {
            val hyperbolicTangentNeuron = HyperbolicTangentNeuron(fuzzyLayer)
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
        val m = 2.0 // m — это весовой коэффициент, который принимает значения из интервала [1, ∞), на практике часто принимают m = 2

        fillUMatrix(u)
        initCenters(u, m)
    }

    // Случайная инициализация коэффициентов матрицы u, выбирая их значения из интервала [0,1] таким образом, чтобы соблюдать условие ∑u = 1
    private fun fillUMatrix(u: Array<DoubleArray>) {
        val r = Random()

        var sum = 0.0
        val vector = DoubleArray(fuzzyLayerSize)
        for (t in trainData.indices) {

            var i = 0
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
        var iteration = 0
        var prevError = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        while (true) {
            // Определить M центров c - ci = ∑(uit) ^ m * xt / ∑(uit) ^ m
            var i = 0
            while (i < fuzzyLayerSize) {
                val fuzzyNeuron = fuzzyLayer[i] as GaussianNeuron
                fuzzyNeuron.center = DoubleArray(fuzzyNeuron.center.size)
                i++
            }

            i = 0
            while (i < fuzzyLayerSize) {
                val fuzzyNeuron = fuzzyLayer[i] as GaussianNeuron
                val center: DoubleArray = fuzzyNeuron.center

                var denominator = 0.0

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
                val fuzzyNeuron = fuzzyLayer[i] as GaussianNeuron
                for (t in trainData.indices) {
                    curError += Math.pow(u[i][t], m) * Math.pow(getEuclideanDistance(fuzzyNeuron.center, trainData[t].Window), 2.0)
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
                val fuzzyNeuron = fuzzyLayer[i] as GaussianNeuron
                for (t in trainData.indices) {

                    val dit: Double = Math.pow(getEuclideanDistance(fuzzyNeuron.center, trainData[t].Window), 2.0)

                    var denominator = 0.0
                    var k = 0
                    while (k < fuzzyLayerSize) {
                        val fuzzyNeuronK = fuzzyLayer[k] as GaussianNeuron
                        val dkt: Double = Math.pow(getEuclideanDistance(fuzzyNeuronK.center, trainData[t].Window), 2.0)
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
        var i = 0
        while (i < fuzzyLayerSize) {
            val fuzzyNeuron = fuzzyLayer[i] as GaussianNeuron
            val norms = DoubleArray(fuzzyLayerSize - 1)

            var j = 0
            var q = 0
            while (j < fuzzyLayerSize) {
                val fuzzyNeuronJ = fuzzyLayer[j] as GaussianNeuron
                if (i != j) {
                    norms[q] = getEuclideanDistance(fuzzyNeuron.center, fuzzyNeuronJ.center)
                    q++
                }
                j++
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

            i++
        }
    }

    private fun backPropagation() {
        var iteration = 0
        var prevError: Double = Double.MAX_VALUE
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
                var k = 0
                while (k < hiddenLayerSize) {
                    val hiddenNeuron = hiddenLayer[k] as AbstractMLPNeuron

                    // δ = ∑(y - d) * (df(u1) / du1) * w * (df(u2) / du2)
                    hiddenNeuron.δ = 0.0
                    for (s in outputLayer.indices) {
                        val outputNeuron = outputLayer[s] as AbstractMLPNeuron
                        hiddenNeuron.δ += outputNeuron.δ * outputNeuron.weights[k]
                    }
                    hiddenNeuron.δ *= hiddenNeuron.activationFunctionDerivative(hiddenNeuron.sum)

                    // ΔW = - η * δ * x
                    for (j in hiddenNeuron.prevLayer.indices) {
                        hiddenNeuron.ΔW[j] = -η * hiddenNeuron.δ * hiddenNeuron.prevLayer[j].value
                    }
                    k++
                }

                k = 0
                while (k < fuzzyLayerSize) {
                    val fuzzyNeuron = fuzzyLayer[k] as GaussianNeuron

                    var sum = 0.0
                    var i = 0
                    while (i < hiddenLayerSize) {
                        val hiddenNeuron = hiddenLayer[i] as AbstractMLPNeuron
                        sum += hiddenNeuron.δ * hiddenNeuron.weights[k] * fuzzyNeuron.value
                        i++
                    }
                    //constant /= hiddenLayer.size

                    fuzzyNeuron.dEdr = 0.0
                    for (j in fuzzyNeuron.center.indices) {
                        fuzzyNeuron.dEdc[j] = sum * (fuzzyNeuron.prevLayer[j].value - fuzzyNeuron.center[j]) / Math.pow(fuzzyNeuron.radius, 2.0)
                        fuzzyNeuron.dEdr += fuzzyNeuron.dEdc[j] / fuzzyNeuron.radius
                    }

                    fuzzyNeuron.dEdr /= fuzzyNeuron.center.size
                    k++
                }

                // Уточнение весов вsходного слоя
                k = 0
                while (k < outputLayerSize) {
                    val outputNeuron = outputLayer[k] as AbstractMLPNeuron
                    for (j in outputNeuron.weights.indices) {
                        outputNeuron.weights[j] += outputNeuron.ΔW[j]
                    }
                    k++
                }

                // Уточнение весов скрытого слоя
                k = 0
                while (k < hiddenLayerSize) {
                    val hiddenNeuron = hiddenLayer[k] as AbstractMLPNeuron
                    for (j in hiddenNeuron.weights.indices) {
                        hiddenNeuron.weights[j] += hiddenNeuron.ΔW[j]
                    }
                    k++
                }

                // Уточнение центра и радиуса нечеткого слоя
                k = 0
                while (k < fuzzyLayerSize) {
                    val fuzzyNeuron = fuzzyLayer[k] as GaussianNeuron
                    for (i in fuzzyNeuron.center.indices) {
                        fuzzyNeuron.center[i] += -η * fuzzyNeuron.dEdc[i]
                    }
                    fuzzyNeuron.radius += -η * fuzzyNeuron.dEdr
                    k++
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