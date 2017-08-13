package k

import k.nets.FMLP
import k.nets.NeuralNetwork

fun main(args: Array<String>) {

    val neuralNetwork: NeuralNetwork = FMLP(
            "temperature.csv", // gold.txt temperature.csv
            80,
            3,
            3,
            6,
            1,
            0.0001,
            10000,
            0.01,
            0.000001,
            10000
    )

    neuralNetwork.prepareData()
    neuralNetwork.buildNetwork()
    neuralNetwork.learn()
    neuralNetwork.test()
}

// TODO: Перейти с Int на Byte. Заменить ArrayList на что-нить попроще.
// TODO: Написать фабрику для создания нейроных сетей с заданными параметрами (в том числи и меняя тип нейронов в скрытом слое, а потом и алгоритмы инициализации)
// TODO: Единичный нейрон в скрытом слое?

//val neuralNetwork: NeuralNetwork = MLP(
//        /* Настройка данных */
//        "gold.txt", // Название файла с данными // gold.txt temperature.csv
//        80, // Процент деления обучающего и тестового набора
//
//        /* Настройка сети */
//        3, // Число входных нейронов = размер скользящего окна
//        6, // Число нейронов скрытого слоя
//        1, // Число выходных нейронов = размер прогноза
//
//        /* Настройка обучения */
//        0.01, // Коэффициент обучения
//        0.000001, // Желаемая минимальная разница погрешностей
//        10000 // Максимальное число итераций обучения
//)