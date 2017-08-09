package k.nets

import k.neurons.*
import k.utils.DataVector
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
) : MLP(dataFileName, trainTestDivide, inputLayerSize, outputLayerSize, hiddenLayerSize, η, errorThresholdBackPropagation, iterationThresholdBackPropagation) {

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
        val m: Int = 2 // m — это весовой коэффициент, который принимает значения из интервала [1, ∞), на практике часто принимают m = 2

        fillUMatrix(u)
        initCenters(u, m)
        calcRadiuses()
    }

    // Случайная инициализация коэффициентов матрицы u, выбирая их значения из интервала [0,1] таким образом, чтобы соблюдать условие ∑u = 1
    private fun fillUMatrix(u: Array<DoubleArray>) {
        val r: Random = Random()

        var i: Int = 0
        var sum: Double = 0.0
        val vector: DoubleArray = kotlin.DoubleArray(trainData.size)
        while (i < fuzzyLayerSize) {

            for (j in trainData.indices) {
                val value: Double = r.nextDouble()
                vector[j] = value
                sum += value
            }

            for (l in vector.indices) {
                vector[l] /= sum // Нормируем значения коэффициентов, таким образом, чтобы они были от 0 до 1 и в сумме давали 1
            }

            sum = 0.0
            i++
        }
    }

    private fun initCenters(u: Array<DoubleArray>, m: Int) {
        var iteration: Int = 0
        var prevError: Double = Double.MAX_VALUE
        var errorDiff: Double

        do {
            // Определить M центров c в соответствии с ∑(u) ^ m * x / ∑(u) ^ m
            var i: Int = 0
            while (i < fuzzyLayerSize) {

                val center: ArrayList<Double> = fuzzyLayer[j].center

                for (t in trainData.indices) {

                    var numerator: Double = 0.0
                    var denominator: Double = 0.0
                    for (j in center.indices) {
                        numerator += u[i][t] * trainData[t].Forecast[j]
                    }

                }





                i++
            }





            iteration++
        } while (errorThresholdFuzzyCMeans < errorDiff && iterationThresholdFuzzyCMeans > iteration)
    }

    private fun calcRadiuses() {

    }
}