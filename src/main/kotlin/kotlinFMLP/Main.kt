package kotlinFMLP

import kotlinFMLP.nets.FMLP
import kotlinFMLP.nets.IFMLP
import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neuralNetworkCriterias.NNCCommand
import kotlinFMLP.neuralNetworkCriterias.TayleDiscrepancyRatioNNC
import kotlinFMLP.neuronFactories.AbstractNeuronFactory
import kotlinFMLP.neuronFactories.LogisticNeuronFactory
import kotlinFMLP.neuronWeightsOptimizers.NWOCommand
import kotlinFMLP.utils.format

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

//    researches.add(SimulatedAnnealingNWO(300.0, 0.9))
//    researches.add(SimulatedAnnealingNWO(300.0, 0.8))

//    researches.add(SimulatedAnnealingNWO(100.0, 0.9))
//    researches.add(SimulatedAnnealingNWO(100.0, 0.8))
//
//    researches.add(SimulatedAnnealingNWO(50.0, 0.9))
//    researches.add(SimulatedAnnealingNWO(50.0, 0.8))

//    researches.add(GeneticNWO(70, 10, 1.0, 0.0))
//    researches.add(GeneticNWO(70, 10, 0.8, 0.2))
//    researches.add(GeneticNWO(70, 10, 0.5, 0.5))
//    researches.add(GeneticNWO(70, 10, 1.0, 0.5))

//    researches.add(GeneticNWO(15, 65, 1.0, 0.0))
//    researches.add(GeneticNWO(15, 65, 0.8, 0.2))
//    researches.add(GeneticNWO(15, 65, 0.5, 0.5))
//    researches.add(GeneticNWO(15, 65, 1.0, 0.5))
//
//    researches.add(GeneticNWO(40, 40, 1.0, 0.0))
//    researches.add(GeneticNWO(40, 40, 0.8, 0.2))
//    researches.add(GeneticNWO(40, 40, 0.5, 0.5))
//    researches.add(GeneticNWO(40, 40, 1.0, 0.5))

//    researches.add(ParticleSwarmNWO(100, 30, 3.0, 2.0, 1.0))
//    researches.add(ParticleSwarmNWO(100, 30, 2.0, 3.0, 1.0))
//    researches.add(ParticleSwarmNWO(100, 30, 2.5, 2.5, 0.7))
//    researches.add(ParticleSwarmNWO(100, 30, 1.5, 4.0, 1.0))

//    researches.add(ParticleSwarmNWO(50, 50, 3.0, 2.0, 1.0))
//    researches.add(ParticleSwarmNWO(50, 50, 2.0, 3.0, 1.0))
//    researches.add(ParticleSwarmNWO(50, 50, 2.5, 2.5, 0.7))
//    researches.add(ParticleSwarmNWO(50, 50, 1.5, 4.0, 1.0))

//    researches.add(ParticleSwarmNWO(30, 100, 3.0, 2.0, 1.0))
//    researches.add(ParticleSwarmNWO(30, 100, 2.0, 3.0, 1.0))
//    researches.add(ParticleSwarmNWO(30, 100, 2.5, 2.5, 0.7))
//    researches.add(ParticleSwarmNWO(30, 100, 1.5, 4.0, 1.0))

//    researches.add(AntColonyNWO(5, 30, 6, 10.0))
//    researches.add(AntColonyNWO(5, 50, 4, 0.1))
//    researches.add(AntColonyNWO(5, 75, 4, 0.1))
//    researches.add(AntColonyNWO(5, 100, 4, 0.1))

//    researches.add(AntColonyNWO(15, 30, 6, 10.0))
//    researches.add(AntColonyNWO(15, 25, 4, 0.1))
//    researches.add(AntColonyNWO(15, 50, 4, 0.1))
//    researches.add(AntColonyNWO(15, 75, 4, 0.1))

//    researches.add(AntColonyNWO(30, 30, 6, 10.0))
//    researches.add(AntColonyNWO(30, 25, 4, 0.1))
//    researches.add(AntColonyNWO(30, 50, 4, 0.1))

//    researches.add(AntColonyNWO(50, 10, 6, 10.0))
//    researches.add(AntColonyNWO(50, 20, 4, 0.1))
//    researches.add(AntColonyNWO(50, 30, 4, 0.1))
//    researches.add(AntColonyNWO(75, 20, 4, 0.1))

    makeResearch(neuralNetwork, researches, 1)
}

fun makeResearch(nn: NeuralNetwork, researches: ArrayList<NWOCommand>, experimentsCount: Int) {
    var initError: Double
    var afterFuzzyLayerInitError = 0.0
    var afterOptimizationError: Double
    var finalError: Double

    nn.prepareData()
    nn.buildNetwork()

    var i = 0
    while (i < experimentsCount) {
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
                println("Эффективность алгоритма: ${(afterFuzzyLayerInitError - afterOptimizationError).format()}")
            } else {
                println("${research}: ${initError.format()} ${afterOptimizationError.format()} ${finalError.format()}")
            }
            nn.clearNetwork()
        }
        nn.shuffleData()
        i++
    }
}