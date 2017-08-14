package k.utils

import java.io.FileNotFoundException
import java.util.*

var dif: Double = 0.0
var min: Double = 0.0

fun readData(trainData: ArrayList<DataVector>, testData: ArrayList<DataVector>, trainTestDivide: Int, dataFileName: String, inputLayerSize: Int, outputLayerSize: Int) {
    val fileIn = ClassLoader.getSystemResourceAsStream("k/datasets/$dataFileName") ?: throw FileNotFoundException(dataFileName)

    val scanner = Scanner(fileIn)
    val readedLines = ArrayList<String>()
    while (scanner.hasNext()) {
        readedLines.add(scanner.nextLine())
    }
    scanner.close()

    val dataValues = ArrayList<Double>()
    var max: Double = Double.MIN_VALUE
    min = Double.MAX_VALUE

    for (i in readedLines.indices) {
        if (i == 0) {
            continue
        }
        val dataString = readedLines[i].split(",")
        val dataValue = dataString[1].toDouble()

        if (dataValue > max)
            max = dataValue
        else if (dataValue < min)
            min = dataValue

        dataValues.add(dataValue)
    }

    dif = max - min

    val trainDataCount = (dataValues.count() - inputLayerSize - outputLayerSize) * trainTestDivide / 100

    var i = 0
    while (i < dataValues.count() - inputLayerSize - outputLayerSize) {
        val dataVector = DataVector(inputLayerSize, outputLayerSize)
        var j = 0

        while (j < inputLayerSize) {
            dataVector.Window[j] = dataValues[i + j]
            j++
        }

        j = 0
        while (j < outputLayerSize) {
            dataVector.Forecast[j] = dataValues[i + inputLayerSize + j]
            j++
        }
        if (i < trainDataCount) {
            trainData.add(dataVector)
        } else {
            testData.add(dataVector)
        }
        i++
    }
}

fun normalized(value: Double): Double {
    return (value - min) / dif
}

fun denormalized(value: Double): Double {
    return value * dif + min
}

fun Double.format(digits: Int) = java.lang.String.format("%.${digits}f", this)

fun getEuclideanDistance(v1: DoubleArray, v2: DoubleArray): Double {
    var sum: Double = 0.0
    for (i in v2.indices) {
        sum += Math.pow(v1[i] - v2[i], 2.0)
    }
    return Math.sqrt(sum)
}