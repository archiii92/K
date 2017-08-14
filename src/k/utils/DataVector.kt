package k.utils

class DataVector(windowSize: Int, forecastSize: Int) {
    val Window: DoubleArray = DoubleArray(windowSize)
    val Forecast: DoubleArray = DoubleArray(forecastSize)
}