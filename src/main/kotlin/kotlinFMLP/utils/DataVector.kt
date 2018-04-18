package kotlinFMLP.utils

class DataVector(windowSize: Int, forecastSize: Int, ForecastDate: String) {
    val Window: DoubleArray = DoubleArray(windowSize)
    val Forecast: DoubleArray = DoubleArray(forecastSize)
    val ForecastDate: String = ForecastDate
}