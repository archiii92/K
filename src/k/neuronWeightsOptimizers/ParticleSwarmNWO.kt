package k.neuronWeightsOptimizers

import k.nets.NeuralNetwork
import k.neurons.AbstractMLPNeuron
import k.utils.format
import k.utils.toFormatString
import java.util.*
import kotlin.collections.ArrayList

class ParticleSwarmNWO(particleCount: Int, val iterationCount: Int, φp: Double, φg: Double, k: Double, override val showLogs: Boolean = false) : NWOCommand {
    val swarm: Swarm = Swarm(particleCount, φp, φg, k)

    override fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
        val beforeError = neuralNetwork.calculateError(neuralNetwork.trainData)
        val weightsChanges = DoubleArray(neuron.weights.size)
        val oldNeuronWeigths = neuron.weights.clone()
        swarm.create(neuron.weights.size)

        var i = 0
        while (i < iterationCount) {
            swarm.calculate(neuron, neuralNetwork)
            i++
        }

        for (j in neuron.weights.indices) {
            weightsChanges[j] = swarm.globalBestPosition[j] - oldNeuronWeigths[j]
            neuron.weights[j] = swarm.globalBestPosition[j]
        }

        swarm.clear()

        val afterError = neuralNetwork.calculateError(neuralNetwork.trainData)

        if (beforeError < afterError) {
            for (j in neuron.weights.indices) {
                neuron.weights[j] = oldNeuronWeigths[j]
            }
            if (showLogs) println("Алгоритм роя частиц не смог найти лучшей комбинации весов для этого нейрона, чем та, которая у него сейчас")
        } else {
            if (showLogs) println("Итог изм весов: ${weightsChanges.toFormatString(4)} Итог улуч ошибки: ${(beforeError - afterError).format(4)}")
        }
    }

    class Swarm(val particleCount: Int, val φp: Double, val φg: Double, val k: Double) {
        val particles: ArrayList<Particle> = ArrayList(particleCount)
        var globalBestValue: Double = Double.MAX_VALUE
        lateinit var globalBestPosition: DoubleArray

        fun create(dimensionSize: Int) {
            var i = 0
            while (i < particleCount) {
                particles.add(Particle(dimensionSize, φp, φg, k))
                i++
            }
            globalBestPosition = DoubleArray(dimensionSize)
        }

        fun calculate(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
            particles.forEach {
                it.calculateCurrentLocal(neuron, neuralNetwork)
                val newError = it.localBestValue
                if (newError < globalBestValue) {
                    globalBestValue = newError
                    globalBestPosition = it.x
                }
            }

            particles.forEach {
                it.calculateVelocity(globalBestPosition)
            }

            particles.forEach {
                it.calculatePosition()
            }
        }

        fun clear() {
            particles.clear()
            globalBestValue = Double.MAX_VALUE
            globalBestPosition = DoubleArray(0)
        }
    }

    class Particle(dimensionSize: Int, val φp: Double, val φg: Double, k: Double) {
        val x: DoubleArray = DoubleArray(dimensionSize)
        val v: DoubleArray = DoubleArray(dimensionSize)

        var localBestValue: Double = Double.MAX_VALUE
        var localBestPosition: DoubleArray = DoubleArray(dimensionSize)

        val φ = φp + φg // φ > 4
        val χ = 2 * k / Math.abs(2 - φ - Math.sqrt(Math.pow(φ, 2.0) - 4 * φ))

        init {
            val r = Random()
            var i = 0
            while (i < dimensionSize) {
                x[i] = r.nextDouble()
                i++
            }
        }

        fun calculateCurrentLocal(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {

            for (j in neuron.weights.indices) {
                neuron.weights[j] = x[j]
            }

            val currentValue = neuralNetwork.calculateError(neuralNetwork.trainData)
            if (localBestValue > currentValue) {
                localBestValue = currentValue
                localBestPosition = x
            }
        }

        fun calculateVelocity(globalBestPosition: DoubleArray) {
            val r = Random()
            for (i in v.indices) {
                val rp = r.nextDouble()
                val rg = r.nextDouble()

                // vi,t+1 = χ [vi,t + φp rp (pi - xi, t) + φg rg (gi - xi, t)] Канонический алгоритм
                v[i] = χ * (v[i] + φp * rp * (localBestPosition[i] - x[i]) + φg * rg * (globalBestPosition[i] - x[i]))
            }
        }

        fun calculatePosition() {
            for (i in x.indices) {
                x[i] = x[i] + v[i]
            }
        }
    }

    override fun toString(): String {
        return "Алгоритм роя частиц"
    }
}