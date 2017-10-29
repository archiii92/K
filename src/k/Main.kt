package k

import k.nets.FMLP
import k.nets.IFMLP
import k.nets.NeuralNetwork
import k.neuralNetworkCriterias.NNCCommand
import k.neuralNetworkCriterias.TayleDiscrepancyRatioNNC
import k.neuronFactories.AbstractNeuronFactory
import k.neuronFactories.LogisticNeuronFactory
import k.neuronWeightsOptimizers.GeneticNWO
import k.neuronWeightsOptimizers.NWOCommand
import k.neuronWeightsOptimizers.ParticleSwarmNWO
import k.neuronWeightsOptimizers.SimulatedAnnealingNWO
import k.utils.format

fun main(args: Array<String>) {

    val neuronFactory: AbstractNeuronFactory = LogisticNeuronFactory()
    val neuralNetworkCriteria: NNCCommand = TayleDiscrepancyRatioNNC()

    val neuralNetwork: NeuralNetwork = FMLP(
            "gold.txt", // gold.txt temperature.csv
            80,
            3,
            9,
            6,
            1,
            0.0000001,
            100,
            0.001,
            0.00001,
            5000,
            3,
            neuronFactory,
            neuralNetworkCriteria
    )

//    val neuralNetwork: NeuralNetwork = MLP(
//            /* Настройка данных */
//            "gold.txt", // Название файла с данными // gold.txt temperature.csv
//            80, // Процент деления обучающего и тестового набора
//
//            /* Настройка сети */
//            3, // Число входных нейронов = размер скользящего окна
//            4, // Число нейронов скрытого слоя
//            1, // Число выходных нейронов = размер прогноза
//
//            /* Настройка обучения */
//            0.001, // Коэффициент обучения
//            0.000001, // Желаемая минимальная разница погрешностей
//            10000, // Максимальное число итераций обучения
//            neuronFactory,
//            neuralNetworkCriteria
//    )

    val researches: ArrayList<NWOCommand> = ArrayList<NWOCommand>()
    researches.add(SimulatedAnnealingNWO(10))
    researches.add(SimulatedAnnealingNWO(50))
    researches.add(SimulatedAnnealingNWO(100))

    researches.add(ParticleSwarmNWO(30, 100, 2.0, 3.0, 1.0))
    researches.add(ParticleSwarmNWO(100, 30, 3.0, 2.0, 1.0))
    researches.add(ParticleSwarmNWO(100, 30, 2.0, 3.0, 0.8))
    researches.add(ParticleSwarmNWO(100, 30, 2.0, 3.0, 1.0))

    researches.add(GeneticNWO(15, 35, 1.0, 0.2))
    researches.add(GeneticNWO(50, 10, 1.0, 0.2))
    researches.add(GeneticNWO(25, 20, 0.8, 0.4))
    researches.add(GeneticNWO(25, 20, 1.0, 0.2))

    makeResearch(neuralNetwork, researches, 1)
}

fun makeResearch(nn: NeuralNetwork, researches: ArrayList<NWOCommand>, experimentsCount: Int){
    var initError: Double
    var afterFuzzyLayerInitError: Double = 0.0
    var afterOptimizationError: Double
    var finalError: Double

    nn.prepareData()
    nn.buildNetwork()

    var i = 0
    while(i < experimentsCount) {
        for (research in researches) {
            initError = nn.calculateError(nn.testData)
            if (nn is IFMLP) {
                nn.initFuzzyLayer()
                afterFuzzyLayerInitError = nn.calculateError(nn.testData)
            }
            nn.optimizeMLPNeuronWeigths(research)
            afterOptimizationError = nn.calculateError(nn.testData)
            nn.learn()
            finalError = nn.calculateError(nn.testData)
            if (nn is IFMLP) {
                println("${research}: ${initError.format()} ${afterFuzzyLayerInitError.format()} ${afterOptimizationError.format()} ${finalError.format()}")
            } else {
                println("${research}: ${initError.format()} ${afterOptimizationError.format()} ${finalError.format()}")
            }
            nn.clearNetwork()
        }
        nn.shuffleData()
        i++
    }
}