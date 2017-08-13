package k.nets

import k.neurons.*
import k.utils.DataVector
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
        iterationThresholdBackPropagation: Int
) : MLP(dataFileName, trainTestDivide, inputLayerSize, hiddenLayerSize, outputLayerSize, η, errorThresholdBackPropagation, iterationThresholdBackPropagation) {

    val fuzzyLayer: ArrayList<GaussianNeuron> = ArrayList(fuzzyLayerSize)

    fun array2dOfDouble(sizeOuter: Int, sizeInner: Int): Array<DoubleArray>
            = Array(sizeOuter) { DoubleArray(sizeInner) }

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
        super.learn()
    }

    override fun calculateOutput(dataVector: DataVector) {
        setInputValue(dataVector)
        calculateFuzzyLayer()
        calculateHiddenLayer()
        calculateOutputLayer()
    }

    private fun calculateFuzzyLayer() {
        fuzzyLayer.forEach { neuron: Neuron -> neuron.calculateState() }
    }

    private fun fuzzyCMeans() {

        //val dataCount: Int = trainData.size
        val u: Array<DoubleArray> = Array(fuzzyLayerSize) { DoubleArray(trainData.size) }
        val m: Double = 2.0 // m — это весовой коэффициент, который принимает значения из интервала [1, ∞), на практике часто принимают m = 2

        fillUMatrix(u)
        initCenters(u, m)
        calcRadiuses()
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

            //u[i] = vector

            sum = 0.0
        }
    }

    private fun initCenters(u: Array<DoubleArray>, m: Double) {
        var iteration: Int = 0
        var prevError: Double = Double.MAX_VALUE
        var errorDiff: Double
        var curError: Double

        do {
            // Определить M центров c - ci = ∑(uit) ^ m * xt / ∑(uit) ^ m
            var i: Int = 0
            while (i < fuzzyLayerSize) {

                val center: DoubleArray = fuzzyLayer[i].center
                //center.clear()

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
                    curError += Math.pow(u[i][t], m) * getEuclideanDistance(fuzzyLayer[i].center, trainData[t].Window)

                }
                i++
            }




            iteration++
        } while (errorThresholdFuzzyCMeans < errorDiff && iterationThresholdFuzzyCMeans > iteration)
    }

    private fun calcRadiuses() {

    }
}