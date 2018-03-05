package kotlinFMLP.neuronWeightsOptimizers

import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neurons.AbstractMLPNeuron
import kotlinFMLP.utils.format
import kotlinFMLP.utils.toFormatString
import java.util.*

class AntColonyNWO(val antCount: Int, val iterationCount: Int, val n: Int, val α: Double, override val showLogs: Boolean = false) : NWOCommand {
    val antColony: AntColony = AntColony(antCount, iterationCount, n, showLogs)

    override fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
        val beforeError = neuralNetwork.calculateError(neuralNetwork.trainData)
        val weightsChanges = DoubleArray(neuron.weights.size)
        val oldNeuronWeights = neuron.weights.clone()
        antColony.create(neuron.weights.size, α)

        antColony.calculate(neuron, neuralNetwork)

        for (j in neuron.weights.indices) {
            weightsChanges[j] = antColony.globalBestPosition[j] - oldNeuronWeights[j]
            neuron.weights[j] = antColony.globalBestPosition[j]
        }

        antColony.clear()

        val afterError = neuralNetwork.calculateError(neuralNetwork.trainData)

        if (beforeError < afterError) {
            for (j in neuron.weights.indices) {
                neuron.weights[j] = oldNeuronWeights[j]
            }
            if (showLogs) println("Алгоритм муравьиной колонии не смог найти лучшей комбинации весов для этого нейрона, чем та, которая у него сейчас")
        } else {
            if (showLogs) println("Итог изм весов: ${weightsChanges.toFormatString(4)} Итог улуч ошибки: ${(beforeError - afterError).format(4)}")
        }
    }

    class AntColony(val antCount: Int, val I: Int, val n: Int, val showLogs: Boolean) {
        val ants: ArrayList<Ant> = ArrayList(antCount)
        var α: Double = 0.0

        var globalBestValue: Double = Double.MAX_VALUE
        lateinit var globalBestPosition: DoubleArray

        var currentBestValue: Double = Double.MAX_VALUE
        lateinit var currentBestPosition: DoubleArray

        fun create(dimensionSize: Int, α: Double) {
            var i = 0
            while (i < antCount) {
                ants.add(Ant(dimensionSize))
                i++
            }
            globalBestPosition = DoubleArray(dimensionSize)
            currentBestPosition = DoubleArray(dimensionSize)
            this.α = α
        }

        fun calculate(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
            var k = 1
            while (k < n) {

                var i = 1
                while (i < k * I) {

                    ants.forEach {
                        it.move(α)
                        it.calculateCurrentValue(neuron, neuralNetwork)
                    }

                    ants.forEach {
                        if (it.value <= currentBestValue) {
                            currentBestValue = it.value
                            for (j in it.position.indices) {
                                currentBestPosition[j] = it.position[j]
                            }
                        }
                    }

                    if (showLogs) println("Итерация: ${k} ${i} Прыжок: ${α} Прошлая ошибка на итерации: ${globalBestValue.format(4)} Текущая ошибка на итерации: ${currentBestValue.format(4)}")

                    if (currentBestValue <= globalBestValue) {
                        globalBestValue = currentBestValue
                        for (j in globalBestPosition.indices) {
                            globalBestPosition[j] = currentBestPosition[j]
                        }
                    }

                    ants.forEach {
                        it.moveToBestPos(globalBestPosition)
                    }

                    i++
                }

                α *= 0.1
                k++
            }
        }

        fun clear() {
            ants.clear()
            globalBestValue = Double.MAX_VALUE
            globalBestPosition = DoubleArray(0)
            currentBestValue = Double.MAX_VALUE
            currentBestPosition = DoubleArray(0)
        }
    }

    class Ant(dimensionSize: Int) {
        var position: DoubleArray = DoubleArray(dimensionSize)
        var value: Double = Double.MAX_VALUE

        val r = Random()

        init {
            var i = 0
            while (i < dimensionSize) {
                position[i] = r.nextDouble()
                i++
            }
        }

        fun move(step: Double) {
            for (i in position.indices) {
                position[i] += r.nextDouble() * 2 * step - step
            }
        }

        fun moveToBestPos(globalBestPosition: DoubleArray) {
            for (i in position.indices) {
                position[i] = globalBestPosition[i]
            }
        }

        fun calculateCurrentValue(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
            for (j in neuron.weights.indices) {
                neuron.weights[j] = position[j]
            }

            value = neuralNetwork.calculateError(neuralNetwork.trainData)
        }
    }

    override fun toString(): String {
        return "Алгоритм муравьиной колонии q = ${antCount}, t = ${iterationCount}, n = ${n}, α = ${α}"
    }
}