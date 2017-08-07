package k.utils

class DataVector(windowSize: Int, forecastSize: Int) {
    val Window: ArrayList<Double> = ArrayList(windowSize)
    val Forecast: ArrayList<Double> = ArrayList(forecastSize)
}