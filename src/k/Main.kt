package k

import k.nets.FMLP
import k.nets.NeuralNetwork
import k.neuronFactories.AbstractNeuronFactory
import k.neuronFactories.LogisticNeuronFactory
import k.neuronWeightsOptimizers.NWOCommand
import k.neuronWeightsOptimizers.SimulatedAnnealingNWO

fun main(args: Array<String>) {

    //val neuronWeightsOptimizer: NWOCommand = SimulatedAnnealingNWO(10)
    val neuronFactory: AbstractNeuronFactory = LogisticNeuronFactory()

    val neuralNetwork: NeuralNetwork = FMLP(
            "gold.txt", // gold.txt temperature.csv
            80,
            3,
            6,
            6,
            1,
            0.0000001,
            300,
            0.001,
            0.00001,
            5000,
            3,
            neuronFactory
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
//            neuronFactory
//    )

    neuralNetwork.prepareData()
    neuralNetwork.buildNetwork()
    neuralNetwork.optimizeMLPNeuronWeigths()
    neuralNetwork.learn()
    neuralNetwork.test()
}

// TODO: Перейти с Double на Float?