package k

import k.nets.FMLP
import k.nets.MLP
import k.nets.NeuralNetwork

fun main(args: Array<String>) {

    val neuralNetwork: NeuralNetwork = FMLP(
            "temperature.csv", // gold.txt temperature.csv
            80,
            3,
            3,
            4,
            1,
            0.0000001,
            300,
            0.005,
            0.0000001,
            10000,
            3
    )

//    val neuralNetwork: NeuralNetwork = MLP(
//        /* Настройка данных */
//        "gold.txt", // Название файла с данными // gold.txt temperature.csv
//        80, // Процент деления обучающего и тестового набора
//
//        /* Настройка сети */
//        3, // Число входных нейронов = размер скользящего окна
//        4, // Число нейронов скрытого слоя                           /// для золота только 4
//        1, // Число выходных нейронов = размер прогноза
//
//        /* Настройка обучения */
//        0.001, // Коэффициент обучения
//        0.000001, // Желаемая минимальная разница погрешностей
//        10000 // Максимальное число итераций обучения
//)

    neuralNetwork.prepareData()
    neuralNetwork.buildNetwork()
    neuralNetwork.learn()
    neuralNetwork.test()
}

// TODO: Перейти с Double на Float?
// TODO: Написать фабрику для создания нейроных сетей с заданными параметрами (в том числи и меняя тип нейронов в скрытом слое, а потом и алгоритмы инициализации)
// TODO: Единичный нейрон в скрытом слое?