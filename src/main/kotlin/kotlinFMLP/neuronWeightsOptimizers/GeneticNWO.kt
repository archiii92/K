package kotlinFMLP.neuronWeightsOptimizers

import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neurons.AbstractMLPNeuron
import kotlinFMLP.utils.format
import kotlinFMLP.utils.toFormatString
import java.util.*
import kotlin.collections.ArrayList

class GeneticNWO(val speciesCount: Int, val iterationCount: Int, val crossPossibility: Double, val mutationPossibility: Double, override val showLogs: Boolean = false) : NWOCommand {
    val population = Population(speciesCount, crossPossibility, mutationPossibility)

    override fun optimizeWeights(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
        val beforeError = neuralNetwork.calculateError(neuralNetwork.trainData)
        val weightsChanges = DoubleArray(neuron.weights.size)
        val oldNeuronWeigths = neuron.weights.clone()
        population.create(neuron.weights.size)

        var i = 0
        while (i < iterationCount) {
            population.nextGeneration(neuron, neuralNetwork)
            i++
        }

        for (j in neuron.weights.indices) {
            weightsChanges[j] = population.species[0].gens[j] - oldNeuronWeigths[j]
            neuron.weights[j] = population.species[0].gens[j]
        }

        population.clear()

        val afterError = neuralNetwork.calculateError(neuralNetwork.trainData)

        if (beforeError < afterError) {
            for (j in neuron.weights.indices) {
                neuron.weights[j] = oldNeuronWeigths[j]
            }
            if (showLogs) println("Генетический алгоритм не смог найти лучшей комбинации весов для этого нейрона, чем та, которая у него сейчас")
        } else {
            if (showLogs) println("Итог изм весов: ${weightsChanges.toFormatString(4)} Итог улуч ошибки: ${(beforeError - afterError).format(4)}")
        }
    }

    class Population(val speciesCount: Int, val crossPossibility: Double, val mutationPossibility: Double) {
        val species: ArrayList<Species> = ArrayList(speciesCount)

        fun create(dimensionSize: Int) {
            var i = 0
            while (i < speciesCount) {
                val sp = Species(dimensionSize)
                sp.primalyInit()
                species.add(sp)
                i++
            }
        }

        fun nextGeneration(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {
            val r = Random()

            val newSP: ArrayList<Species> = ArrayList(2 * speciesCount)

            species.forEach {
                val random = r.nextDouble()
                if (crossPossibility > random) {
                    val speciesN = r.nextInt(species.size)
                    newSP.addAll(it.cross(this.species[speciesN]))
                }
            }

            species.addAll(newSP)

            species.forEach {
                val random = r.nextDouble()
                if (mutationPossibility > random) {
                    it.mutation()
                }
            }

            species.forEach {
                it.calculateValue(neuron, neuralNetwork)
            }

            species.sortBy { it.value }

            species.subList(speciesCount, species.size).clear()
        }

        fun clear() {
            species.clear()
        }
    }

    class Species(val dimensionSize: Int) {
        val gens: DoubleArray = DoubleArray(dimensionSize)
        var value: Double = 0.0

        fun primalyInit() {
            val r = Random()
            var i = 0
            while (i < dimensionSize) {
                gens[i] = r.nextDouble()
                i++
            }
        }

        fun calculateValue(neuron: AbstractMLPNeuron, neuralNetwork: NeuralNetwork) {

            for (j in neuron.weights.indices) {
                neuron.weights[j] = gens[j]
            }

            value = neuralNetwork.calculateError(neuralNetwork.trainData)
        }

        fun cross(species: Species): ArrayList<Species> {
            val r = Random()

            val gensN = r.nextInt(gens.size)

            val sp1 = Species(this.gens.size)
            val sp2 = Species(this.gens.size)

            for (i in gens.indices) {
                if (i <= gensN) {
                    sp1.gens[i] = this.gens[i]
                    sp2.gens[i] = species.gens[i]
                } else {
                    sp1.gens[i] = species.gens[i]
                    sp2.gens[i] = this.gens[i]
                }
            }
            val result = ArrayList<Species>(2)
            result.add(sp1)
            result.add(sp2)
            return result
        }

        fun mutation() {
            val r = Random()

            val gensN = r.nextInt(gens.size)

            this.gens[gensN] = r.nextDouble()
        }
    }

    override fun toString(): String {
        return "Генетический алгоритм q = ${speciesCount}, t = ${iterationCount}, p = ${crossPossibility} и m = ${mutationPossibility}"
    }
}


