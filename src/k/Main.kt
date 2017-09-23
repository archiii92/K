package k

import k.nets.FMLP
import k.nets.MLP
import k.nets.NeuralNetwork
import k.neuronFactories.AbstractNeuronFactory
import k.neuronFactories.LogisticNeuronFactory
import k.neuronWeightsOptimizers.GeneticNWO
import k.neuronWeightsOptimizers.NWOCommand

fun main(args: Array<String>) {

    val neuronFactory: AbstractNeuronFactory = LogisticNeuronFactory()
    //val neuronWeightsOptimizer: NWOCommand = SimulatedAnnealingNWO(100)
    //val neuronWeightsOptimizer: NWOCommand = ParticleSwarmNWO(100, 30, 2.0, 3.0, 1.0)
    val neuronWeightsOptimizer: NWOCommand = GeneticNWO(25, 20, 1.0, 0.2)

//    val neuralNetwork: NeuralNetwork = FMLP(
//            "gold.txt", // gold.txt temperature.csv
//            80,
//            3,
//            9,
//            6,
//            1,
//            0.0000001,
//            100,
//            0.001,
//            0.00001,
//            5000,
//            3,
//            neuronFactory
//    )

    val neuralNetwork: NeuralNetwork = MLP(
            /* Настройка данных */
            "gold.txt", // Название файла с данными // gold.txt temperature.csv
            80, // Процент деления обучающего и тестового набора

            /* Настройка сети */
            3, // Число входных нейронов = размер скользящего окна
            4, // Число нейронов скрытого слоя
            1, // Число выходных нейронов = размер прогноза

            /* Настройка обучения */
            0.001, // Коэффициент обучения
            0.000001, // Желаемая минимальная разница погрешностей
            10000, // Максимальное число итераций обучения
            neuronFactory
    )

    neuralNetwork.prepareData()
    neuralNetwork.buildNetwork()
    //neuralNetwork.optimizeMLPNeuronWeigths(neuronWeightsOptimizer)
    neuralNetwork.learn()
    neuralNetwork.test()
}

// TODO: Перейти с Double на Float?