package k

import k.nets.MLP
import k.nets.NeuralNetwork

fun main(args: Array<String>) {
    val neuralNetwork: NeuralNetwork = MLP(
            /* Настройка данных */
            "gold.txt", // Название файла с данными // gold.txt temperature.csv
            80, // Процент деления обучающего и тестового набора

            /* Настройка сети */
            3, // Число входных нейронов = размер скользящего окна
            1, // Число выходных нейронов = размер прогноза
            6, // Число нейронов скрытого слоя

            /* Настройка обучения */
            0.01, // Коэффициент обучения
            0.000001, // Желаемая минимальная разница погрешностей
            10000 // Максимальное число итераций обучения
    )

    neuralNetwork.prepareData()
    neuralNetwork.buildNetwork()
    neuralNetwork.learn()
    neuralNetwork.test()
}