package kotlinFMLP.controllers

import java.util.*

class MainModel {
    fun makeForecast(forecastSettings: ForecastSettings): DoubleArray {
        println("Параметры вызова: ${forecastSettings.fileName} ${forecastSettings.selectedNetwork} ${forecastSettings.selectedAlgorithm}")
        println("Параметры вызова: ${forecastSettings.initialTemperature} ${forecastSettings.warmingKeepPercent}")

        val array = DoubleArray(5)
        val r = Random()

        for (i in array.indices) {
            val value: Double = r.nextDouble()
            array[i] = value
        }

        return array
    }
}