package kotlinFMLP

import kotlinFMLP.nets.FMLP
import kotlinFMLP.nets.IFMLP
import kotlinFMLP.nets.NeuralNetwork
import kotlinFMLP.neuralNetworkCriterias.NNCCommand
import kotlinFMLP.neuralNetworkCriterias.TayleDiscrepancyRatioNNC
import kotlinFMLP.neuronFactories.AbstractNeuronFactory
import kotlinFMLP.neuronFactories.LogisticNeuronFactory
import kotlinFMLP.neuronWeightsOptimizers.AntColonyNWO
import kotlinFMLP.neuronWeightsOptimizers.NWOCommand
import kotlinFMLP.utils.format

fun main(args: Array<String>) {

    val neuronFactory: AbstractNeuronFactory = LogisticNeuronFactory()
    val neuralNetworkCriteria: NNCCommand = TayleDiscrepancyRatioNNC()

    val neuralNetwork: NeuralNetwork = FMLP(
            "gold.txt", // gold.txt temperature.csv
            80,
            1,
            6,
            9,
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
//            1, // Число входных нейронов = размер скользящего окна
//            6, // Число нейронов скрытого слоя
//            1, // Число выходных нейронов = размер прогноза
//
//            /* Настройка обучения */
//            0.001, // Коэффициент обучения
//            0.00001, // Желаемая минимальная разница погрешностей
//            5000, // Максимальное число итераций обучения
//            neuronFactory,
//            neuralNetworkCriteria
//    )

    val researches: ArrayList<NWOCommand> = ArrayList<NWOCommand>()

//    researches.add(SimulatedAnnealingNWO(300.0, 0.9))
//    researches.add(SimulatedAnnealingNWO(300.0, 0.7))
//
//    researches.add(SimulatedAnnealingNWO(150.0, 0.9))
//    researches.add(SimulatedAnnealingNWO(150.0, 0.7))
//
//    researches.add(SimulatedAnnealingNWO(75.0, 0.9))  // !!! 50
//    researches.add(SimulatedAnnealingNWO(75.0, 0.7))
//
//    researches.add(SimulatedAnnealingNWO(35.0, 0.9))
//    researches.add(SimulatedAnnealingNWO(35.0, 0.7))

//    researches.add(GeneticNWO(70, 10, 1.0, 0.0))
//    researches.add(GeneticNWO(70, 10, 0.8, 0.2))
//    researches.add(GeneticNWO(70, 10, 0.5, 0.5))
//    researches.add(GeneticNWO(70, 10, 1.0, 0.5))
//
//    researches.add(GeneticNWO(15, 65, 1.0, 0.0))
//    researches.add(GeneticNWO(15, 65, 0.8, 0.2))
//    researches.add(GeneticNWO(15, 65, 0.5, 0.5))
//    researches.add(GeneticNWO(15, 65, 1.0, 0.5))  // !!!
//
//
//    researches.add(GeneticNWO(40, 40, 1.0, 0.0))
//    researches.add(GeneticNWO(40, 40, 0.8, 0.2))
//    researches.add(GeneticNWO(40, 40, 0.5, 0.5))
//    researches.add(GeneticNWO(40, 40, 1.0, 0.5))

//
//    researches.add(ParticleSwarmNWO(100, 30, 3.0, 2.0, 1.0))
//    researches.add(ParticleSwarmNWO(100, 30, 2.0, 3.0, 1.0))  // !!!
//
//
//    researches.add(ParticleSwarmNWO(100, 30, 2.5, 2.5, 0.7))
//    researches.add(ParticleSwarmNWO(100, 30, 1.5, 4.0, 1.0))
//
//    researches.add(ParticleSwarmNWO(50, 50, 3.0, 2.0, 1.0))
//    researches.add(ParticleSwarmNWO(50, 50, 2.0, 3.0, 1.0))
//    researches.add(ParticleSwarmNWO(50, 50, 2.5, 2.5, 0.7))
//    researches.add(ParticleSwarmNWO(50, 50, 1.5, 4.0, 1.0))
//
//    researches.add(ParticleSwarmNWO(30, 100, 3.0, 2.0, 1.0))
//    researches.add(ParticleSwarmNWO(30, 100, 2.0, 3.0, 1.0))
//    researches.add(ParticleSwarmNWO(30, 100, 2.5, 2.5, 0.7))
//    researches.add(ParticleSwarmNWO(30, 100, 1.5, 4.0, 1.0))

    researches.add(AntColonyNWO(5, 50, 6, 10.0))
    researches.add(AntColonyNWO(5, 50, 4, 0.1))
    researches.add(AntColonyNWO(5, 100, 4, 0.1))
    researches.add(AntColonyNWO(5, 200, 4, 0.1))

    researches.add(AntColonyNWO(25, 20, 6, 10.0))
    researches.add(AntColonyNWO(25, 20, 4, 0.1))
    researches.add(AntColonyNWO(25, 40, 4, 0.1))
    researches.add(AntColonyNWO(25, 60, 4, 0.1))

    //researches.add(AntColonyNWO(30, 25, 4, 0.1))  // !!!

    researches.add(AntColonyNWO(50, 10, 6, 10.0))
    researches.add(AntColonyNWO(50, 10, 4, 0.1))
    researches.add(AntColonyNWO(50, 30, 4, 0.1))
    researches.add(AntColonyNWO(75, 20, 4, 0.1))

    makeResearch(neuralNetwork, researches, 3)
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
                //println("Эффективность алгоритма: ${(afterFuzzyLayerInitError - afterOptimizationError).format()}")
            } else {
                println("${research}: ${initError.format()} ${afterOptimizationError.format()} ${finalError.format()}")
            }
            nn.clearNetwork()
        }
        nn.shuffleData()
        i++
    }
}