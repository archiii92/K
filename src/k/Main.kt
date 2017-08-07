package k

import k.nets.MLP
import k.nets.NeuralNetwork

fun main(args: Array<String>) {
    val mlp: NeuralNetwork = MLP(
            /* Настройка данных */
            "gold.txt", // Название файла с данными // gold.txt temperature.csv
            80, // Процент деления обучающего и тестового набора

            /* Настройка сети */
            3, // Число входных нейронов = размер скользящего окна
            1, // Число выходных нейронов = размер прогноза
            6, // Число нейронов скрытого слоя

            /* Настройка обучения */
            0.01, // Коэффициент обучения
            5e-6, // Желаемая погрешность
            10000          // Число итераций обучения
    )
    mlp.prepareData()
    mlp.buildNetwork()
    mlp.learn()
    mlp.test()
}